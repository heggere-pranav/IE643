import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import requests
import zipfile
import numpy as np
import cv2
import random
from sklearn.metrics import f1_score, average_precision_score
from transformers import BertTokenizer, BertModel, DetrImageProcessor, DetrForObjectDetection, pipeline
from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from answer_mappings import answer_mapping

class VQAWithAnnotation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.img_enc = self._build_image_encoder(config["image_encoder"])
        self.multi_modal_combine = nn.Linear(self.text_encoder.config.hidden_size + self.img_enc_out_dim, 512)
        self.classifier = nn.Linear(512, config["num_answers"])
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
        self.vqa_pipeline = pipeline("vqa", model="dandelin/vilt-b32-finetuned-vqa")
        self.answer_mapping = self._create_answer_mapping()
        self.device = config["device"]

    def _build_image_encoder(self, config):
        if config["type"] == "resnet":
            resnet = models.resnet50(pretrained=config["params"]["pretrained"])
            self.img_enc_out_dim = resnet.fc.in_features
            modules = list(resnet.children())[:-1]
            return nn.Sequential(*modules)
        else:
            raise ValueError("Unsupported image encoder type")

    def _create_answer_mapping(self):
        flat_mapping = {}
        for key, variations in answer_mapping.items():
            for variation in variations:
                flat_mapping[variation] = key
        return flat_mapping

    def forward(self, img, question):
        img = self._ensure_rgb(img)
        quoted_phrases = self._extract_quoted_phrases(question)
        if quoted_phrases:
            answer = quoted_phrases[0]
        else:
            answer = self.vqa_pipeline(img, question)[0]['answer']
        answer = self._map_answer(answer)
        bboxes, labels = self._get_bounding_boxes(img, answer)
        return answer, bboxes, labels

    def _map_answer(self, answer):
        return self.answer_mapping.get(answer.lower(), answer)

    def _get_bounding_boxes(self, img, answer):
        detr_inputs = self.detr_processor(images=img, return_tensors="pt")
        detr_outputs = self.detr_model(**detr_inputs)
        target_label = self.config["answer_to_target_label"].get(answer, None)
        bboxes, labels = [], []
        if target_label is not None:
            probs = torch.softmax(detr_outputs.logits, -1)[0]
            for idx, label in enumerate(probs.argmax(-1)):
                if label == target_label and probs[idx, label] > 0.5:
                    bbox = detr_outputs.pred_boxes[0][idx].detach().cpu().numpy()
                    bboxes.append(bbox)
                    labels.append(answer)
        return bboxes, labels

    def get_segmented_masks(self, img, target_label):
        img_tensor = transforms.ToTensor()(img)
        outputs = self.maskrcnn_model([img_tensor])[0]
        masks, labels = outputs['masks'], outputs['labels']
        return [(masks[i, 0].detach().cpu().numpy(), target_label) for i in range(len(masks)) if labels[i].item() == target_label]

    def highlight_segmented_objects(self, img, segmented_objects, answer):
        img_np = np.array(img)
        color = [random.randint(0, 255) for _ in range(3)]
        for mask, _ in segmented_objects:
            img_np = self._overlay_mask(img_np, mask, color)
            img_np = self._draw_contours(img_np, mask, color, answer)
        return Image.fromarray(img_np)

    @staticmethod
    def _ensure_rgb(img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

    @staticmethod
    def _overlay_mask(img_np, mask, color):
        mask_overlay = np.zeros_like(img_np)
        mask_overlay[mask > 0.5] = color
        return cv2.addWeighted(img_np, 1, mask_overlay, 0.5, 0)

    @staticmethod
    def _draw_contours(img_np, mask, color, answer):
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contours, -1, color, 2)
        if contours:
            x, y = contours[0][0][0]
            cv2.putText(img_np, f"Object: {answer}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_np

    @staticmethod
    def _extract_quoted_phrases(text):
        import re
        return re.findall(r'"(.*?)"', text)

    def unsupervised_finetune(self, loader, epochs=5, lr=1e-4):
        self.train()
        opt = optim.Adam(self.parameters(), lr=lr)

        all_targets, all_preds = [], []
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, questions, targets in loader:
                imgs = imgs.to(self.device)
                opt.zero_grad()

                feats = self.img_enc(imgs).squeeze()
                batch_size = feats.size(0)
                perm_idx = torch.randperm(batch_size)
                pos_pairs = feats
                neg_pairs = feats[perm_idx]

                pos_sim = nn.functional.cosine_similarity(pos_pairs, feats)
                neg_sim = nn.functional.cosine_similarity(neg_pairs, feats)

                loss = -torch.mean(torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))))
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                total_loss += loss.item()

                # For evaluation metrics
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(pos_sim.detach().cpu().numpy() > 0.5)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(loader):.4f}")

        # Calculate metrics
        f1 = f1_score(all_targets, all_preds, average='weighted')
        mAP = average_precision_score(all_targets, all_preds, average='weighted')
        iou = self.calculate_iou(all_preds, all_targets)

        print(f"Final Training Loss: {total_loss / (epochs * len(loader)):.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        print(f"Intersection over Union (IoU): {iou:.4f}")

    @staticmethod
    def calculate_iou(preds, targets):
        intersection = np.logical_and(preds, targets).sum()
        union = np.logical_or(preds, targets).sum()
        if union == 0:
            return 0.0
        return intersection / union

# MS COCO Dataset Handling
def download_coco():
    img_url = "http://images.cocodataset.org/zips/train2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    img_zip = "train2017.zip"
    ann_zip = "annotations_trainval2017.zip"
    if not os.path.exists(img_zip):
        print("Downloading COCO images...")
        r = requests.get(img_url, stream=True)
        with open(img_zip, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(img_zip, 'r') as zip_ref:
            zip_ref.extractall("./")

    if not os.path.exists(ann_zip):
        print("Downloading COCO annotations...")
        r = requests.get(ann_url, stream=True)
        with open(ann_zip, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall("./")

# COCO Dataset Loader Class
class COCOData(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        super().__init__()
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgToAnns.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        question = anns[0]['caption']
        answer = anns[0]['category_id']

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, question, answer

# Define Transformations and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Main Training Script
def main():
    download_coco()

    img_dir = "train2017"
    ann_file = "annotations/instances_train2017.json"

    dataset = COCOData(img_dir, ann_file, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    config = {
        "image_encoder": {"type": "resnet", "params": {"pretrained": True}},
        "num_answers": 91,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "answer_to_target_label": {str(i): i for i in range(91)}
    }
    model = VQAWithAnnotation(config).to(config["device"])

    model.unsupervised_finetune(loader, epochs=5, lr=1e-4)

if __name__ == "__main__":
    main()

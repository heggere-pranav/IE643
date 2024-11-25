import torch
import torch.nn as nn
from PIL import Image
from transformers import BertTokenizer, BertModel, DetrImageProcessor, DetrForObjectDetection, pipeline
import numpy as np
import cv2
from torchvision import models, transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import random


class VQAWithAnnotation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.image_encoder = self._build_image_encoder(self.config["image_encoder"])
        self.multi_modal_combine = nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder_out_dim, 512)
        self.classifier = nn.Linear(512, self.config["num_answers"])
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
        self.vqa_pipeline = pipeline("vqa", model="dandelin/vilt-b32-finetuned-vqa")
        self.answer_mapping = self._create_answer_mapping()

    def _build_image_encoder(self, config):
        if config["type"] == "resnet":
            resnet = models.resnet50(pretrained=config["params"]["pretrained"])
            self.image_encoder_out_dim = resnet.fc.in_features
            modules = list(resnet.children())[:-1]  # Remove the fully connected layer
            return nn.Sequential(*modules)
        else:
            raise ValueError("Unsupported image encoder type")

    def _create_answer_mapping(self):
        # Create list of possible word variations and map them to the exact form
        answer_mapping = {
            "man": ["men", "man", "person", "people"],
            "bicycle": ["bicycles", "bicycle", "bike", "bikes"],
            "car": ["cars", "car"],
            "motorcycle": ["motorcycles", "motorcycle", "bike", "bikes"],
            "airplane": ["airplanes", "plane", "planes", "airplane"],
            "bus": ["buses", "bus"],
            "train": ["trains", "train"],
            "truck": ["trucks", "truck"],
            "boat": ["boats", "boat"],
            "traffic light": ["traffic lights", "traffic light"],
            "fire hydrant": ["fire hydrants", "hydrant", "fire hydrant"],
            "stop sign": ["stop signs", "stop sign"],
            "parking meter": ["parking meters", "parking meter"],
            "bench": ["benches", "bench"],
            "bird": ["birds", "bird"],
            "cat": ["cats", "cat"],
            "dog": ["dogs", "dog"],
            "horse": ["horses", "horse"],
            "sheep": ["sheep"],
            "cow": ["cows", "cow"],
            "elephant": ["elephants", "elephant"],
            "bear": ["bears", "bear"],
            "zebra": ["zebras", "zebra"],
            "giraffe": ["giraffes", "giraffe"],
            "backpack": ["backpacks", "backpack"],
            "umbrella": ["umbrellas", "umbrella"],
            "handbag": ["handbags", "handbag"],
            "tie": ["ties", "tie"],
            "suitcase": ["suitcases", "suitcase"],
            "frisbee": ["frisbees", "frisbee"],
            "skis": ["skis", "ski"],
            "snowboard": ["snowboards", "snowboard"],
            "sports ball": ["sports balls", "sports ball", "ball", "balls"],
            "kite": ["kites", "kite"],
            "baseball bat": ["baseball bats", "bat", "baseball bat"],
            "baseball glove": ["baseball gloves", "glove", "baseball glove"],
            "skateboard": ["skateboards", "skateboard"],
            "surfboard": ["surfboards", "surfboard"],
            "tennis racket": ["tennis rackets", "racket", "tennis racket"],
            "bottle": ["bottles", "bottle"],
            "wine glass": ["wine glasses", "glass", "wine glass"],
            "cup": ["cups", "cup"],
            "fork": ["forks", "fork"],
            "knife": ["knives", "knife"],
            "spoon": ["spoons", "spoon"],
            "bowl": ["bowls", "bowl"],
            "banana": ["bananas", "banana"],
            "apple": ["apples", "apple"],
            "sandwich": ["sandwiches", "sandwich"],
            "orange": ["oranges", "orange"],
            "broccoli": ["broccoli"],
            "carrot": ["carrots", "carrot"],
            "hot dog": ["hot dogs", "hotdog", "hot dog"],
            "pizza": ["pizzas", "pizza"],
            "donut": ["donuts", "doughnuts", "donut", "doughnut"],
            "cake": ["cakes", "cake"],
            "chair": ["chairs", "chair"],
            "couch": ["couches", "sofa", "couch"],
            "potted plant": ["potted plants", "plant", "potted plant"],
            "bed": ["beds", "bed"],
            "dining table": ["dining tables", "table", "dining table"],
            "toilet": ["toilets", "toilet"],
            "tv": ["television", "tv", "tv set"],
            "laptop": ["laptops", "laptop"],
            "mouse": ["mice", "mouse"],
            "remote": ["remotes", "remote"],
            "keyboard": ["keyboards", "keyboard"],
            "cell phone": ["cell phones", "phone", "cellphone", "cell phone"],
            "microwave": ["microwaves", "microwave"],
            "oven": ["ovens", "oven"],
            "toaster": ["toasters", "toaster"],
            "sink": ["sinks", "sink"],
            "refrigerator": ["refrigerators", "fridge", "refrigerator"],
            "book": ["books", "book"],
            "clock": ["clocks", "clock"],
            "vase": ["vases", "vase"],
            "scissors": ["scissors"],
            "teddy bear": ["teddy bears", "teddy", "teddy bear"],
            "hair drier": ["hair dryers", "hair drier", "hair dryer"],
            "toothbrush": ["toothbrushes", "brush", "toothbrush"]
        }
        # Flatten the variations into a mapping
        flat_mapping = {}
        for key, variations in answer_mapping.items():
            for variation in variations:
                flat_mapping[variation] = key
        return flat_mapping

    def forward(self, image, question):
        image = self._ensure_rgb(image)
        # Check if the question contains any phrase in double quotes
        quoted_phrases = self._extract_quoted_phrases(question)
        if quoted_phrases:
            answer = quoted_phrases[0]  # Use the first quoted phrase as the answer
        else:
            # Use VQA pipeline to get the answer
            answer = self.vqa_pipeline(image, question)[0]['answer']
        # Map answer to the target label, accounting for different forms
        answer = self._map_answer(answer)
        # Get bounding boxes and labels from DETR
        bboxes, labels = self._get_bounding_boxes(image, answer)
        return answer, bboxes, labels

    def _map_answer(self, answer):
        return self.answer_mapping.get(answer.lower(), answer)

    def _get_bounding_boxes(self, image, answer):
        detr_inputs = self.detr_processor(images=image, return_tensors="pt")
        detr_outputs = self.detr_model(**detr_inputs)
        target_label = self.config["answer_to_target_label"].get(answer, None)
        bboxes, labels = [], []
        if target_label is not None:
            # Apply softmax to get probabilities and filter by threshold
            probs = torch.softmax(detr_outputs.logits, -1)[0]
            for idx, label in enumerate(probs.argmax(-1)):
                if label == target_label and probs[idx, label] > 0.5:  # Threshold to filter uncertain predictions
                    bbox = detr_outputs.pred_boxes[0][idx].detach().cpu().numpy()
                    bboxes.append(bbox)
                    labels.append(answer)
        return bboxes, labels

    def get_segmented_masks(self, image, target_label):
        """Generates pixel-level masks for detected objects using Mask R-CNN, only for objects matching the target label."""
        image_tensor = transforms.ToTensor()(image)
        outputs = self.maskrcnn_model([image_tensor])[0]
        masks, labels = outputs['masks'], outputs['labels']
        # Extract only masks that match the target label
        return [(masks[i, 0].detach().cpu().numpy(), target_label) for i in range(len(masks)) if labels[i].item() == target_label]

    def highlight_segmented_objects(self, image, segmented_objects, answer):
        """Highlights only the segmented objects matching the VQA answer with a randomly generated color."""
        image_np = np.array(image)
        color = [random.randint(0, 255) for _ in range(3)]  # Random color for answer objects
        for mask, _ in segmented_objects:
            # Overlay mask with color
            image_np = self._overlay_mask(image_np, mask, color)
            # Draw contours and add label text
            image_np = self._draw_contours(image_np, mask, color, answer)
        return Image.fromarray(image_np)

    @staticmethod
    def _ensure_rgb(image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    @staticmethod
    def _overlay_mask(image_np, mask, color):
        mask_overlay = np.zeros_like(image_np)
        mask_overlay[mask > 0.5] = color
        return cv2.addWeighted(image_np, 1, mask_overlay, 0.5, 0)

    @staticmethod
    def _draw_contours(image_np, mask, color, answer):
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_np, contours, -1, color, 2)
        if contours:
            x, y = contours[0][0][0]
            cv2.putText(image_np, f"Object: {answer}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image_np

    @staticmethod
    def _extract_quoted_phrases(text):
        import re
        return re.findall(r'"(.*?)"', text)

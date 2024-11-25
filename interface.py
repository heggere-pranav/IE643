import streamlit as st
from PIL import Image
from vqa_with_annotation import VQAWithAnnotation

# Configuration for the VQA model
config = {
    "image_encoder": {
        "type": "resnet",
        "params": {
            "pretrained": True,
            "num_layers": 101  # Updated number of layers to improve feature extraction
        }
    },
    "num_answers": 2000,  # Increased number of possible answers
    "answer_to_target_label": {
        "man": 1,
        "bicycle": 2,        "car": 3,        "motorcycle": 4,        "airplane": 5,        "bus": 6,        "train": 7,        "truck": 8,        "boat": 9,        "traffic light": 10,        "fire hydrant": 11,        "stop sign": 13,        "parking meter": 14,        "bench": 15,        "bird": 16,        "cat": 17,        "dog": 18,        "horse": 19,        "sheep": 20,        "cow": 21,        "elephant": 22,        "bear": 23,        "zebra": 24,        "giraffe": 25,        "backpack": 27,        "umbrella": 28,        "handbag": 31,        "tie": 32,        "suitcase": 33,        "frisbee": 34,        "skis": 35,        "snowboard": 36,        "sports ball": 37,        "kite": 38,        "baseball bat": 39,        "baseball glove": 40,        "skateboard": 41,        "surfboard": 42,        "tennis racket": 43,        "bottle": 44,        "wine glass": 46,        "cup": 47,        "fork": 48,        "knife": 49,        "spoon": 50,        "bowl": 51,        "banana": 52,        "apple": 53,        "sandwich": 54,        "orange": 55,        "broccoli": 56,        "carrot": 57,        "hot dog": 58,        "pizza": 59,        "donut": 60,        "cake": 61,        "chair": 62,        "couch": 63,        "potted plant": 64,        "bed": 65,        "dining table": 67,        "toilet": 70,        "tv": 72,        "laptop": 73,        "mouse": 74,        "remote": 75,        "keyboard": 76,        "cell phone": 77,        "microwave": 78,        "oven": 79,        "toaster": 80,        "sink": 81,        "refrigerator": 82,        "book": 84,        "clock": 85,        "vase": 86,        "scissors": 87,        "teddy bear": 88,        "hair drier": 89,        "toothbrush": 90  
    }
}

# Initialize the model
model = VQAWithAnnotation(config)

# Streamlit app
st.set_page_config(page_title="VQA Live Demo", layout="wide")
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextInput > div > input {
            font-size: 18px;
            padding: 10px;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 VQA Annotator Live Demonstration Interface")
st.write("This application allows you to ask questions about images and get visual answers. Simply upload an image and enter your question!")

# Upload image
uploaded_file = st.file_uploader("📂 Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Ask a question
    st.markdown("### Ask any Question About the Image")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer") and question:
        # Process the image and question
        with st.spinner("Processing your question... please wait."):
            answer, bboxes, labels = model(image, question)
            target_label = config["answer_to_target_label"].get(answer)

        # Display answer and results
        st.markdown(f"### Answer: **{answer}**")
        if target_label is not None:
            segmented_objects = model.get_segmented_masks(image, target_label)
            highlighted_image = model.highlight_segmented_objects(image, segmented_objects, answer)

            st.image(highlighted_image, caption='Highlighted Image', use_column_width=True)
        else:
            st.warning(f"No segmentation available for the object: {answer}")

# Footer
st.markdown(
    """
    <hr style="height:2px;border-width:0;color:gray;background-color:gray">
    <p style="text-align:center">Developed by EigenLords.</p>
    """,
    unsafe_allow_html=True
)

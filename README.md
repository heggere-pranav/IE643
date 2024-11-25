# Visual Question Answering with Object Detection and Pixel-Level Annotation

This repository presents a comprehensive solution for integrating Visual Question Answering (VQA) with object detection and pixel-level annotation. The project leverages advanced deep learning models to process images and answer user queries, providing annotated visual outputs.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Running the Application](#running-the-application)
6. [Usage](#usage)
7. [Performance Metrics](#performance-metrics)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

This project aims to develop a technique that utilizes VQA models for object detection and pixel-level annotation through unsupervised fine-tuning. The key components include:

- **Data Collection:** Utilizing the MS COCO dataset for training and evaluation.
- **Model Exploration:** Implementing deep learning models for VQA and unsupervised fine-tuning.
- **Training and Fine-Tuning:** Adapting pre-trained models to the specific task.
- **Performance Metrics:** Establishing metrics to evaluate object detection and annotation quality.
- **Interface Development:** Creating a user-friendly interface for live demonstrations.

## Prerequisites

Before setting up the project, ensure the following are installed:

- **Python 3.8 or higher**
- **pip**: Python package installer
- **Git**: Version control system

## Installation

Follow these steps to set up the project environment:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/heggere-pranav/IE643.git
   cd IE643
   ```

2. **Create a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv vqa_env
   ```

3. **Activate the Virtual Environment:**

   - On Windows:

     ```bash
     vqa_env\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source vqa_env/bin/activate
     ```

4. **Install Required Packages:**

   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, manually install the following packages:

   ```bash
   pip install torch torchvision transformers streamlit numpy opencv-python pillow
   ```

## Dataset Preparation

The project utilizes the [MS COCO dataset](https://cocodataset.org/) for training and evaluation. Follow these steps to prepare the dataset:

1. **Download the Dataset:**

   - **Images:** Download the 2017 Train, Validation, and Test images from the [COCO dataset download page](https://cocodataset.org/#download).

   - **Annotations:** Download the corresponding annotation files for the 2017 dataset.

2. **Organize the Dataset:**

   Create a directory structure as follows:

   ```
   dataset/
   ├── images/
   │   ├── train2017/
   │   ├── val2017/
   │   └── test2017/
   └── annotations/
       ├── instances_train2017.json
       ├── instances_val2017.json
       └── image_info_test2017.json
   ```

3. **Update Configuration:**

   Ensure that the dataset paths in your configuration files or scripts point to the correct locations of the images and annotations.

## Running the Application

With the environment set up and the dataset prepared, you can run the application as follows:

1. **Start the Streamlit Interface:**

   Navigate to the project directory and run:

   ```bash
   streamlit run app.py
   ```

   This command will launch the Streamlit application in your default web browser.

2. **Upload an Image:**

   Use the interface to upload an image in `.jpg`, `.jpeg`, or `.png` format.

3. **Ask a Question:**

   Enter a question related to the uploaded image (e.g., "What is the person holding?") and submit.

4. **View Results:**

   The application will display the answer along with the image annotated to highlight relevant objects.

## Usage

The application provides an interactive interface for users to:

- **Upload Images:** Select images for analysis.
- **Pose Questions:** Ask questions about the content of the images.
- **Receive Annotated Outputs:** View images with highlighted objects corresponding to the answers.

## Performance Metrics

To evaluate the performance of the model, the following metrics are considered:

- **Accuracy:** Measures the correctness of the VQA responses.
- **Intersection over Union (IoU):** Assesses the quality of the segmentation masks.
- **Processing Time:** Evaluates the efficiency of the model in generating responses.

## Contributing

Contributions to enhance the project are welcome. Please fork the repository and submit a pull request with your proposed changes.

*Note: Ensure that the paths and configurations in your environment match those specified in this README to avoid any issues during setup and execution.* 

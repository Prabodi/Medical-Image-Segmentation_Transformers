# Medical Image Segmentation - Transformers

Vision Transformers for Glioma Classification Using T1 Magnetic Resonance Images

This repository contains a brain tumor classification and segmentation model developed for glioma classification using T1-weighted magnetic resonance images (MRIs). The model utilizes Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) to achieve high accuracy in classifying glioma tumors into different categories.

Project Overview

Glioma, a common form of brain cancer, is typically diagnosed using medical imaging, including magnetic resonance imaging (MRI). This project aims to build an efficient model for classifying glioma tumors based on T1-weighted MRI scans, leveraging the power of Vision Transformers (ViTs) and CNNs.

The key highlights of the project include:

Classification and Segmentation: A dual-purpose model to classify glioma types and segment the tumor region.

Vision Transformers: Using ViTs for image feature extraction and classification, achieving superior accuracy compared to traditional CNNs.

Fine-tuning: The model was fine-tuned by adjusting image pixel intensities for better classification performance.

Tools: The project was developed using Python, with key libraries like TensorFlow and PyTorch for deep learning, along with specialized ViT and CNN architectures.


Features

Glioma Classification: Classifies gliomas into multiple categories (e.g., benign, malignant).

Tumor Segmentation: Segments the tumor area from the MRI images.

Vision Transformer (ViT) Model: Uses ViTs for image classification tasks, improving performance over conventional CNN approaches.

Preprocessing: Image pixel intensities are adjusted and normalized for improved model accuracy.

Evaluation Metrics: The model is evaluated using metrics such as accuracy, F1 score, and Dice coefficient for segmentation.


Installation

1. Clone the repository:

git clone https://github.com/Prabodi/Medical-Image-Segmentation_Transformers.git

2. Install required dependencies: Create a virtual environment (optional) and install the dependencies using pip:

pip install -r requirements.txt


3. Required libraries:

TensorFlow / PyTorch

NumPy

Matplotlib

OpenCV

scikit-learn

Pillow




Usage

1. Data Preparation:

Download the MRI dataset (e.g., from Kaggle or any other source).

Ensure the data is organized with images labeled for training and testing.



2. Training the Model:

Use the provided Python scripts to train the Vision Transformer and CNN models on the dataset.

Example command for training:


python train.py --data_path /path/to/mri_images --model vit


3. Evaluating the Model:

After training, evaluate the model's performance using:


python evaluate.py --model vit --test_data /path/to/test_images


4. Segmentation:

You can use the model for tumor segmentation using the script segment.py:


python segment.py --input_image /path/to/test_image --model vit

Results

The Vision Transformer model demonstrated superior accuracy in classifying glioma tumors when compared to traditional CNN-based approaches. The results show a significant improvement in precision and recall, particularly when fine-tuning the pixel intensities of the MRI images.

Accuracy: 95%

F1 Score: 0.92

Dice Coefficient (Segmentation): 0.88

Acknowledgements

This project leverages MRI datasets such as the BraTS 2021 dataset for glioma classification and segmentation. Special thanks to the researchers and contributors who made this dataset available for public use.

Contact

For any questions, feel free to reach out to prabodi.ruwanthika@gmail.com

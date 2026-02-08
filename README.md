💄 Face Makeup Detection using CNN
📌 Project Description

This project is a Deep Learning based Image Classification system that detects whether a face has Makeup or No Makeup using a Convolutional Neural Network (CNN).

The model is trained on a facial image dataset and predicts makeup presence from new images.

🎯 Objectives

To build a CNN model for image classification

To classify facial images into:

Makeup

No Makeup

To understand real-world Deep Learning workflow

To deploy project using GitHub

🧠 Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

📂 Dataset

Dataset Source: Kaggle

🔗 https://www.kaggle.com/datasets/ananysharma/face-makeup-detection-dataset

Dataset contains two classes:

with_makeup

no_makeup

⚙️ Project Workflow

Load Dataset using ImageDataGenerator

Apply Data Augmentation

Build CNN Model

Train Model

Evaluate Accuracy

Save Model

Predict New Images

🏗️ CNN Model Architecture

Conv2D + MaxPooling

Conv2D + MaxPooling

Conv2D + MaxPooling

Flatten Layer

Dense Layer

Dropout Layer

Output Layer (Sigmoid)

📊 Model Training Result

Training Accuracy ≈ 58%

Validation Accuracy ≈ 52%

Accuracy graph is included in the repository.

📁 Project Files
dataset/
python train.py
predict.py
accuracy_graph.png
requirements.txt

▶️ How to Run Project
Install Requirements
pip install -r requirements.txt

Train Model
python train.py

Predict Image
python predict.py

💡 GitHub Repository

🔗 https://github.com/shreyakadam-m/Face-Makeup-Detection

👩‍💻 Author

Shreya Kadam

# 🌙 AI-Based Malayalam Handwritten Document Quality Assessment System

<p align="center">
  <img src="https://img.shields.io/badge/AI-Document%20Quality%20Assessment-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  An AI-powered system designed to evaluate the visual quality of Malayalam handwritten documents using image processing and deep learning.
</p>

---

## ✨ Project Overview

Handwritten documents can vary significantly in terms of clarity, sharpness, contrast, brightness, and overall appearance.

This project focuses on developing an **AI-based Malayalam Handwritten Document Quality Assessment System** that analyzes handwritten document images and classifies their quality.

The system combines image preprocessing techniques and deep learning models to assess document quality and provide an automated evaluation.

## 🎯 Objectives

- Evaluate the visual quality of Malayalam handwritten documents.
- Apply image preprocessing techniques to improve image quality.
- Develop deep learning models for document quality classification.
- Compare different CNN architectures.
- Identify an effective model for document quality assessment.
- Provide an interactive interface for uploading and analyzing documents.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Models | CNN, ResNet50, DenseNet, EfficientNet, MobileNet, Xception |
| Interface | Streamlit |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |

## ⚙️ System Workflow

1. **Image Input** – Upload a Malayalam handwritten document.
2. **Preprocessing** – Prepare the image for analysis.
3. **Feature Learning** – Extract important visual patterns using deep learning.
4. **Model Prediction** – Analyze the document using the trained model.
5. **Quality Assessment** – Generate the predicted document quality category.
6. **Result Display** – Present the assessment through the Streamlit interface.

## 🧠 Image Preprocessing

The preprocessing stage prepares handwritten document images for deep learning analysis.

Techniques explored include:

- Grayscale conversion
- Image resizing
- Padding
- Gaussian blur
- Thresholding
- Edge detection
- Morphological operations

## 🤖 Deep Learning Models

Multiple CNN-based architectures were explored and evaluated:

- CNN
- ResNet50
- Fine-Tuned ResNet50
- DenseNet
- EfficientNet
- MobileNet
- Xception

**Selected Model:** Fine-Tuned ResNet50

The fine-tuned ResNet50 model achieved the best performance among the evaluated models in this project.

## 📊 Model Evaluation

The models are evaluated using classification performance metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

*Exact evaluation scores can be added here after final verification.*

## 🖥️ Application

The project includes a Streamlit-based interface that allows users to upload handwritten document images and view the quality assessment results.

**Application features:**
- Image upload
- Document preprocessing
- AI-based quality prediction
- Quality category display
- Interactive interface

## 📁 Project Structure

```text
Malayalam-Handwritten-Document-Quality-Assessment/
│
├── app.py
├── predict.py
├── read_image.py
├── preprocess_dataset.py
├── split_dataset.py
│
├── resnet50_document_quality_model.keras
├── resnet50_finetuned_document_quality_model.keras
│
├── training_scripts/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above is a simplified overview. Your actual repository may contain additional files.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Abiramasree/Malayalam-Handwritten-Document-Quality-Assessment.git
```

### 2. Navigate to the Project Directory

```bash
cd Malayalam-Handwritten-Document-Quality-Assessment
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📌 Dataset

The project uses a dataset of Malayalam handwritten document images categorized according to document quality.

The dataset is organized into quality categories:

- Good
- Average
- Poor

The dataset is not included in this public repository due to its size.

## 🔮 Future Enhancements

- Improve model performance through further experimentation.
- Expand the Malayalam handwritten document dataset.
- Explore advanced deep learning architectures.
- Improve prediction reliability on unseen document images.
- Extend the system to support additional Indian languages.

## 👩‍💻 Developer

**Abiramasree K**

B.Tech Computer Science and Engineering

SRM University–AP

<p>
  <a href="https://github.com/Abiramasree">
    <img src="https://img.shields.io/badge/GitHub-Abiramasree-181717?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.linkedin.com/in/abiramasree-k-335003325/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" />
  </a>
</p>

---

<p align="center">
  <i>Turning handwritten documents into measurable insights through AI.</i>
</p>

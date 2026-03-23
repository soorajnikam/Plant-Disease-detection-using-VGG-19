#  Tomato Plant Disease Detection Using CNN

##  Project Overview

This project focuses on **automated detection and classification of tomato plant diseases** using **Convolutional Neural Networks (CNNs)** and image processing techniques.

The system is designed to help farmers **identify diseases at an early stage** through leaf image analysis, reducing dependency on manual inspection and improving agricultural productivity.

A **Streamlit-based web application** was developed to provide an easy-to-use interface for real-time disease prediction.

---

##  Key Features

*  Image-based disease detection from tomato leaves
*  CNN model for classification of plant diseases
*  Early-stage disease identification
*  Feature extraction (color, texture, edges)
*  User-friendly **Streamlit web app**
*  Fast and automated predictions

---

##  System Architecture

1. **Image Input**

   * Upload leaf images via Streamlit UI

2. **Preprocessing**

   * Image resizing
   * Noise removal
   * Normalization

3. **Image Segmentation**

   * Isolate infected regions from leaf

4. **Feature Extraction (CNN)**

   * Color patterns
   * Texture analysis
   * Edge detection

5. **Model Architecture**

   * 2 Convolutional Layers
   * 2 Pooling Layers
   * Fully Connected Layer

6. **Classification**

   * Predict disease category
   * Output label + confidence score

7. **Visualization**

   * Display prediction results in Streamlit dashboard

---

##  Technologies Used

| Category         | Technologies                       |
| ---------------- | ---------------------------------- |
| Programming      | Python                             |
| Deep Learning    | CNN (TensorFlow / Keras / PyTorch) |
| Image Processing | OpenCV                             |
| Web App          | Streamlit                          |
| Data Handling    | NumPy, Pandas                      |

---

##  Model Details

* Model Type: Convolutional Neural Network (CNN)
* Layers:

  * 2 Convolution Layers
  * 2 Pooling Layers
  * Dense (Fully Connected Layer)
* Input: Tomato leaf images
* Output: Disease classification

---

##  Project Structure

```bash
plant-disease-detection/
│
├── data/                  # Dataset (leaf images)
├── models/                # Trained CNN model
├── notebooks/             # Experimentation
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── model.py
│   ├── predict.py
│
├── app/
│   └── app.py             # Streamlit application
├── results/               # Outputs & metrics
├── README.md
└── requirements.txt
```

---

##  Installation & Setup

###  Clone Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run Streamlit App

```bash
streamlit run app/app.py
```

---

##  How It Works

1. Upload a tomato leaf image
2. System preprocesses and segments the image
3. CNN model extracts features
4. Disease is classified and displayed instantly

---

##  Problem Solved

*  Manual disease detection is time-consuming
*  Requires expert knowledge
*  Difficult for farmers in remote areas

 This system provides:

* Fast and accurate detection
* Easy accessibility via web interface
* Early diagnosis to prevent crop loss

---

##  Future Improvements

*  Mobile application deployment
*  Multi-crop disease detection
*  Cloud-based scalable system
*  Integration with advanced deep learning models (ResNet, EfficientNet)

---

##  Publication

 **Published in IEEE Conference**

* Conference: *International Conference on Computational Intelligence for Green and Sustainable Technologies (ICCIGST 2024)*
*  Location: Vijayawada, India
*  Date: 18–19 July 2024
*  DOI: 10.1109/ICCIGST60741.2024.10717462

---

##  Conclusion

This project demonstrates how **deep learning and computer vision** can significantly improve **precision agriculture**, enabling farmers to detect diseases early and take preventive action effectively.

---

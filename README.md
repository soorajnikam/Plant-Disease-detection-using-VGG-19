#  Tomato Plant Disease Detection Using VGG-19

> **Published at IEEE ICCIGST 2024** — International Conference on Computational Intelligence for Green and Sustainable Technologies  
>  Vijayawada, India &nbsp;|&nbsp; 📅 18–19 July 2024 &nbsp;|&nbsp; 🔗 [DOI: 10.1109/ICCIGST60741.2024.10717462](https://doi.org/10.1109/ICCIGST60741.2024.10717462)

---

##  Overview

This project presents an automated classification system for detecting **tomato leaf diseases** using **Deep Convolutional Neural Networks (DCNN)** and **Transfer Learning with VGG-19**. The system leverages **HSV color space segmentation** to isolate diseased leaf regions before classification, significantly improving accuracy and reducing training time.

The work was trained and validated on the **PlantVillage dataset** (16,000+ tomato leaf images across 9 disease categories) and deployed as a **Streamlit web application** for real-time predictions.

---



##  Disease Categories (PlantVillage Dataset)

The model classifies tomato leaves into **9 disease types** plus healthy:

| # | Disease |
|---|---|
| 1 | Yellow Leaf Curl Virus |
| 2 | Two-Spotted Spider Mites |
| 3 | Bacterial Spot |
| 4 | Septoria Leaf Spot |
| 5 | Mosaic Virus |
| 6 | Leaf Mold |
| 7 | Target Spot |
| 8 | Early Blight |
| 9 | Late Blight |
| 10 | Healthy |

>  **Note on class imbalance**: The Yellow Leaf Curl Virus class has 3,000+ images while the Mosaic Virus class has only 373 images. Balanced class weighting and up-sampling techniques are applied to handle this.

---

##  System Architecture

```
Input Image (Tomato Leaf)
        ↓
  Image Resizing (224×224 or 256×256 px)
        ↓
  HSV Color Space Segmentation
  (Extract leaf regions → Black background)
        ↓
  VGG-19 (Transfer Learning)
  ├── 16 Convolutional Layers
  ├── Max-Pooling Layers
  └── 3 Fully Connected Layers
        ↓
  Softmax Output Layer
        ↓
  Disease Classification + Confidence Score
        ↓
  Streamlit Web Dashboard
```

---

##  Methodology

### 1. Image Preprocessing & Segmentation

Leaf images are converted from **RGB to HSV color space** before training. The HSV space separates color information from intensity, making it more efficient for isolating green leaf regions.

- **Hue (H) threshold**: 0.003 – 0.50
- **Saturation (S) threshold**: 0.15 – 0.80
- Output: Diseased leaf regions isolated against a **pure black background**

This segmentation step:
- Improves classification accuracy
- Reduces training time compared to non-segmented images
- Minimises computational load on the neural network

### 2. VGG-19 with Transfer Learning

The VGG-19 architecture consists of **19 layers** total:
- 16 Convolutional Layers — for feature extraction (color, texture, edges)
- 3 Fully Connected Layers — for disease classification

**Training configuration:**
- Activation (hidden layers): ReLU
- Activation (output layer): Softmax
- Loss function: Categorical Cross-Entropy
- Pooling: Max-Pooling (to reduce features and prevent overfitting)
- Transfer learning type: Parameter-based

**Dataset split:**
| Split | Percentage |
|---|---|
| Training | 52.5% (75% of 70%) |
| Validation | 17.5% (25% of 70%) |
| Testing | 30% |

---

##  Evaluation Metrics

The model is evaluated using the following standard metrics:

- **Accuracy** (RP_accuracy)
- **Precision** (RP_precision)
- **Recall** (RP_recall)
- **F1-Score** (RP_F1−score)
- **Specificity** (SPE)
- **Sensitivity** (SEN)
- **Positive Predictive Value** (PPV)
- Confusion Matrix analysis

---

##  Technologies Used

| Category | Technology |
|---|---|
| Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Image Processing | OpenCV (HSV segmentation) |
| Pre-trained Model | VGG-19 |
| Data Handling | NumPy, Pandas |
| Web Application | Streamlit |
| Dataset | PlantVillage (16,000+ images) |

---

##  Project Structure

```
plant-disease-detection/
│
├── data/                      # PlantVillage dataset (leaf images)
│   ├── raw/                   # Original RGB images
│   └── segmented/             # HSV-segmented images (black background)
│
├── models/
│   └── vgg19_tomato.h5        # Trained VGG-19 model weights
│
├── notebooks/
│   └── experimentation.ipynb  # Model training & evaluation experiments
│
├── src/
│   ├── preprocessing.py       # Image resizing & HSV segmentation
│   ├── segmentation.py        # HSV color space extraction
│   ├── model.py               # VGG-19 architecture & transfer learning
│   └── predict.py             # Inference pipeline
│
├── app/
│   └── app.py                 # Streamlit web application
│
├── results/                   # Confusion matrices, accuracy plots
├── README.md
└── requirements.txt
```

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app/app.py
```

---

##  How It Works

1. **Upload** a tomato leaf image via the Streamlit interface
2. **Preprocessing** — image is resized to 224×224 px
3. **Segmentation** — leaf region is isolated using HSV thresholding; background set to black
4. **Feature extraction** — VGG-19 convolutional layers extract color, texture, and edge features
5. **Classification** — fully connected layers output a disease label with confidence score
6. **Display** — result shown instantly on the Streamlit dashboard

---

## 🌍 Problem Statement

Traditional disease detection in tomato farming:
- Requires domain expertise not available to remote/rural farmers
- Is time-consuming and prone to human error
- Often results in late-stage detection, increasing crop losses

This system provides:
-  Fast, automated detection accessible via web browser
-  Early-stage diagnosis to enable timely treatment
-  No domain expertise required from the end user

---

##  Future Work

-  **Mobile application** deployment for on-field use
-  **Multi-crop** disease detection (extending beyond tomatoes)
-  **Cloud-based** scalable deployment
-  Integration with advanced architectures: **ResNet, EfficientNet**
-  **IoT sensor integration** for real-time field monitoring and treatment automation

---


---

##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

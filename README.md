# Ensemble Deep Learning for Histopathological Breast Cancer Detection

This repository contains the implementation of our research on breast cancer classification using ensemble deep learning approaches on the BreaKHis dataset, as presented in our publication.

**Publication**: [Ensemble Deep Learning for Histopathological Breast Cancer Detection](https://www.medrxiv.org/content/10.1101/2025.08.12.25333539v1)  
**DOI**: https://doi.org/10.1101/2025.08.12.25333539  
**Publication Date**: August 12, 2025

##  Abstract

Breast cancer remains one of the leading causes of mortality among women worldwide, and early and accurate diagnosis is essential for effective treatment. In this study, we propose an ensemble deep learning approach for classifying histopathological images of breast cancer using the BreaKHis dataset. Two state-of-the-art convolutional neural network architectures, ResNet50 and DenseNet121, were fine-tuned and combined through multiple ensemble strategies, including stacking with logistic regression, XGBoost, and hard/soft voting. The models were trained on an 80/20 train-validation split, preserving the distribution of benign and malignant classes across all magnification levels (40X, 100X, 200X, and 400X). Experimental results show that the hard voting ensemble achieved the highest accuracy of 98.61%, closely followed by the XGBoost ensemble with an accuracy of 98.55%, both outperforming individual models. These findings highlight the effectiveness of ensemble deep learning in improving classification performance for breast cancer histopathology and suggest its potential for aiding pathologists in clinical decision-making.

##  Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet50 (Single) | 97.82% | 97.85% | 97.82% | 97.82% |
| DenseNet121 (Single) | 98.12% | 98.15% | 98.12% | 98.12% |
| Logistic Regression Stacking | 98.42% | 98.45% | 98.42% | 98.42% |
| XGBoost Ensemble | 98.55% | 98.58% | 98.55% | 98.55% |
| Hard Voting Ensemble | **98.61%** | **98.64%** | **98.61%** | **98.61%** |

##  Project Structure
project-root/
│
├── models/
│ ├── best_resnet50.keras # Best ResNet50 model
│ ├── best_densenet121.keras # Best DenseNet121 model
│ ├── breakhis_resnet_model.keras # Final ResNet50 model
│ └── breakhis_densenet_model.keras # Final DenseNet121 model
│
├── datasets/
│ └── BreakHis_split/
│ ├── train/ # Training data
│ └── val/ # Validation data
│
├── ensemble.py # Main implementation code
└── README.md # This file

text

##  Model Architectures

### 1. Fine-Tuned ResNet50
- **Base Model**: ResNet50 with ImageNet weights
- **Fine-tuning**: Last 30 layers unfrozen for training
- **Custom Head**: GlobalAveragePooling2D → Dense(512, ReLU) → BatchNorm → Dropout(0.5) → Dense(2, Softmax)

### 2. Fine-Tuned DenseNet121
- **Base Model**: DenseNet121 with ImageNet weights
- **Fine-tuning**: Last 30 layers unfrozen for training
- **Custom Head**: GlobalAveragePooling2D → Dense(512, ReLU) → BatchNorm → Dropout(0.5) → Dense(2, Softmax)

### 3. Ensemble Strategies
- **Stacking with Logistic Regression**
- **XGBoost Ensemble**
- **Hard/Soft Voting Classifier**

## 📊 Dataset

The BreaKHis dataset contains histopathological images of breast tumors:
- **Classes**: Benign vs. Malignant
- **Magnification Levels**: 40X, 100X, 200X, 400X
- **Total Images**: 7,909 (2,480 benign, 5,429 malignant)
- **Split**: 80% training, 20% validation

##  Installation & Requirements

```bash
pip install tensorflow keras scikit-learn matplotlib seaborn numpy xgboost
Usage
python
python ensemble.py
Model Availability
Due to the sensitive nature of the trained models and intellectual property protection, the actual trained model files are not publicly hosted in this repository.

The complete source code for training, evaluation, and ensemble implementation is provided, allowing researchers to replicate our results exactly.

If you require access to the pre-trained models for academic collaboration or research verification, please contact me directly.

👨 Author
Alireza Rahi

📧 Email: alireza.rahi@outlook.com

💼 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/

 GitHub: https://github.com/AlirezaRahi

📄 License
All Rights Reserved.

Copyright (c) 2025 Alireza Rahi

Unauthorized access, use, modification, or distribution of this software is strictly prohibited without explicit written permission from the copyright holder.
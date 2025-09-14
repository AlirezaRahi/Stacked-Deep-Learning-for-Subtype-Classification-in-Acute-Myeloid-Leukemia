# Ensemble Deep Learning for AML Subtype Classification

This repository contains the implementation of our research on AML subtype classification using ensemble deep learning approaches on microarray expression data.

**Publication**:  
**DOI**:
**Publication Date**:

##  Abstract

Acute Myeloid Leukemia (AML) is a genetically heterogeneous disease with distinct molecular subtypes that require precise classification for optimal treatment selection. In this study, we propose an ensemble deep learning approach for classifying AML subtypes using gene expression microarray data. Three neural network architectures (1D CNN, Dense Network, and Transformer) were trained and combined through stacking ensemble strategies with XGBoost as the meta-learner. The models were trained using stratified 5-fold cross-validation on a balanced dataset of AML samples. Experimental results demonstrate that our stacking ensemble approach achieves robust performance across all AML subtypes, with particularly strong results for inv(16) and t(8;21) subtypes. These findings highlight the effectiveness of ensemble deep learning in improving classification performance for AML subtyping and suggest its potential for aiding hematopathologists in clinical decision-making.

##  Key Results

| Subtype | Precision | Recall | F1-Score | Support |
|---------|----------|--------|----------|---------|
| AML_CK | 1.00 | 0.71 | 0.83 | 7 |
| AML_FLT3_ITD | 0.54 | 0.88 | 0.67 | 8 |
| AML_inv_16 | 1.00 | 1.00 | 1.00 | 6 |
| AML_M3_APL | 0.88 | 1.00 | 0.93 | 7 |
| AML_NK | 0.78 | 0.88 | 0.82 | 8 |
| AML_other | 0.67 | 0.25 | 0.36 | 8 |
| AML_t_8_21 | 1.00 | 1.00 | 1.00 | 7 |
| **Overall** | **-** | **-** | **-**** | **51** |



##  Model Architectures

### 1. 1D Convolutional Neural Network
- **Architecture**: Conv1D(128) → BatchNorm → MaxPool → Conv1D(256) → GlobalAveragePooling → Dense(256) → Dropout → Dense(128) → Dropout → Output
- **Activation**: ReLU for hidden layers, Softmax for output

### 2. Dense Network
- **Architecture**: Flatten → Dense(1024) → BatchNorm → Dropout → Dense(512) → BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Output
- **Activation**: ReLU for hidden layers, Softmax for output

### 3. Transformer Network
- **Architecture**: Two transformer blocks with multi-head attention → GlobalAveragePooling → Dropout → Output
- **Attention Heads**: 4, Head Size: 256

### 4. Ensemble Strategy
- **Stacking with XGBoost**: Predictions from base models used as features for XGBoost meta-learner

## 📊 Dataset

The dataset contains gene expression profiles of AML samples:
- **Subtypes**: AML_CK, AML_FLT3_ITD, AML_inv_16, AML_M3_APL, AML_NK, AML_other, AML_t_8_21
- **Samples**: 51 total samples (balanced across subtypes)
- **Features**: 3000 most informative genes selected through ANOVA F-test
- **Preprocessing**: Standard scaling and feature selection

Dataset available at: https://zenodo.org/records/16999485

##  Installation & Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas xgboost joblib
```

## Usage

```bash
python aml_ensemble.py
```

## Model Availability

Due to the sensitive nature of the trained models and intellectual property protection, the actual trained model files are not publicly hosted in this repository.

The complete source code for training, evaluation, and ensemble implementation is provided, allowing researchers to replicate our results exactly.

If you require access to the pre-trained models for academic collaboration or research verification, please contact me directly.

## 👨 Author

Alireza Rahi

📧 Email: alireza.rahi@outlook.com

💼 LinkedIn: https://www.linkedin.com/in/alireza-rahi-6938b4154/

🌐 GitHub: https://github.com/AlirezaRahi

## 📄 License

All Rights Reserved.

Copyright (c) 2025 Alireza Rahi

Unauthorized access, use, modification, or distribution of this software is strictly prohibited without explicit written permission from the copyright holder.

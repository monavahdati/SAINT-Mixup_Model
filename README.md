# SAINT+Mixup for BNPL Credit Classification 💳🧠

This repository implements a Transformer-based tabular classifier (SAINT) enhanced with Mixup augmentation for robust prediction in a Buy Now Pay Later (BNPL) credit system.

---

## 🧠 Architecture

### SAINTEncoder
- Projects input features to embedding space
- Processes through TransformerEncoder layers

### SAINTMixup
- Combines SAINT encoder with:
  - MLP layers + BatchNorm + Dropout
  - Softmax output over 2 classes

---

## 🔁 Mixup Training
- Mixes input pairs:  
  `x' = λ * x₁ + (1-λ) * x₂`
- Loss is calculated from both targets using `lam`

---

## 📊 Metrics & Visualization

- **Accuracy, Precision, Recall, F1, AUC**
- **KS Statistic**
- **Confusion Matrix**
- **Gain & Lift Curves**
- **SHAP Summary Plot**

---

## 📈 Visualizations Included

- KS Curve
- ROC Curve
- Precision-Recall Curve
- Training Accuracy / Loss
- Confusion Matrix Heatmap
- Gain / Lift Chart

---

## 🚀 Run the Project

```python
# Mount drive
drive.mount('/content/drive')

# Load and preprocess data
# Train SAINTMixup model
# Evaluate & visualize results
# Export predictions

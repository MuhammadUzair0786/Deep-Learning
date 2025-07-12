# 🏦 ANN Bank Customer Churn Prediction

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer will churn (leave the bank) based on their profile and transaction data.

---

## 🚀 Project Workflow

1. **Load Data:** Import the bank customer churn dataset using Pandas.
2. **Data Preprocessing:**
   - Split the data into features (`x`) and target (`y`).
   - Scale the input features using StandardScaler for optimal ANN performance.
   - Split the data into training and testing sets.
3. **Model Building:**
   - Build a Sequential ANN model using Keras with multiple Dense layers and ReLU activations.
   - Use a sigmoid activation in the output layer for binary classification.
   - Compile the model with `binary_crossentropy` loss and `adam` optimizer.
4. **Model Training:**
   - Train the model on the training data for a set number of epochs and batch size.
   - Optionally, use validation data and EarlyStopping for better generalization.
5. **Prediction & Evaluation:**
   - Predict churn on both training and test sets.
   - Convert predicted probabilities to binary class labels (0 or 1) using a 0.5 threshold.
   - Evaluate model performance using accuracy score.
6. **Single Prediction:**
   - Predict churn for new/unseen customer data.

---

## 🧠 Key Concepts

- **Feature Scaling:** Ensures all input features contribute equally to the ANN.
- **ANN Architecture:** Multiple hidden layers with ReLU activation for learning complex patterns.
- **Binary Classification:** Output layer uses sigmoid activation for churn prediction (0 = No, 1 = Yes).
- **Model Evaluation:** Accuracy is used to assess model performance.

---

## 💻 Tech Stack

- Python
- Pandas, Numpy
- scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

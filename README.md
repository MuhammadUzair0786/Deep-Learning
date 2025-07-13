# 🏦 ANN Bank Customer Churn Prediction

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer will churn (leave the bank) based on their profile and transaction data.

---

## 🚀 Project Workflow

1. **Load Data:**  
   - Import the bank customer churn dataset using Pandas.

2. **Data Preprocessing:**  
   - Split the data into features (`x`) and target (`y`).
   - Scale the input features using StandardScaler for optimal ANN performance.
   - Split the data into training and testing sets (80/20 split).

3. **Model Building:**  
   - Build a Sequential ANN model using Keras with multiple Dense layers, Batch Normalization, and L2 regularization to reduce overfitting.
   - Use a sigmoid activation in the output layer for binary classification.
   - Compile the model with `binary_crossentropy` loss and `adam` optimizer.

4. **Model Training:**  
   - Train the model on the training data for up to 50 epochs with a batch size of 100.
   - Use validation data and EarlyStopping to prevent overfitting.

5. **Prediction & Evaluation:**  
   - Predict churn on both training and test sets.
   - Convert predicted probabilities to binary class labels (0 or 1) using a 0.5 threshold.
   - Evaluate model performance using accuracy score and confusion matrix.
   - Visualize confusion matrices using seaborn heatmaps.

6. **Single Prediction:**  
   - Predict churn for new/unseen customer data by providing a scaled feature vector.

---

## 🧠 Key Concepts

- **Feature Scaling:** Ensures all input features contribute equally to the ANN.
- **ANN Architecture:** Multiple hidden layers with ReLU activation, Batch Normalization, and L2 regularization for learning complex patterns and reducing overfitting.
- **Binary Classification:** Output layer uses sigmoid activation for churn prediction (0 = No, 1 = Yes).
- **Model Evaluation:** Accuracy and confusion matrix are used to assess model performance.
- **Early Stopping:** Prevents overfitting by stopping training when validation performance stops improving.

---


## 💻 Tech Stack

- Python
- Pandas, Numpy
- scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## 📌 How to Run

1. Clone the repository and navigate to the project folder.
2. Ensure all required libraries are installed (`pip install -r requirements.txt`).
3. Open `ANN-Bank_Customer_Churn_Prediction.ipynb` in Jupyter Notebook.
4. Run each cell step by step to preprocess data, train the model, and evaluate results.

---

This notebook is a practical guide for beginners to understand and apply deep learning for customer churn prediction
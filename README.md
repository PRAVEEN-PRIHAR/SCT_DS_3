
# 📊 Bank Marketing Prediction using Decision Tree & Random Forest

This project involves training and evaluating machine learning models to predict whether a customer will subscribe to a bank's term deposit based on their demographic and behavioral data. It uses the **Bank Direct Marketing Dataset** and applies techniques such as **decision trees**, **random forests**, and **SMOTE** for class balancing.

## 📁 Project Structure

- **bank-direct-marketing.csv** – Input dataset.
- **Bank Marketing Task3.ipynb** – Jupyter Notebook containing data processing, model training, visualization, and evaluation.

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Seaborn & Matplotlib

## 🔍 Key Steps

1. **Data Preprocessing**  
   - Dropped irrelevant columns (e.g., `duration`)
   - Encoded categorical variables using one-hot encoding
   - Encoded target variable (`y`: yes → 1, no → 0)

2. **Model Training**  
   - Trained a Decision Tree Classifier (`max_depth=5`)
   - Trained a Random Forest Classifier (`n_estimators=100`, `max_depth=10`)

3. **Model Evaluation**  
   - Evaluated using accuracy, confusion matrix, and classification report
   - Applied **SMOTE** to balance the dataset and improve model performance on minority class

## ✅ Key Findings

- **Baseline Decision Tree** achieved reasonable interpretability but moderate performance due to class imbalance.
- **Random Forest** significantly improved accuracy and robustness over the decision tree.
- **SMOTE + Random Forest** further improved recall and F1-score, showing enhanced performance for the minority class (customers who subscribed).

## 📉 Sample Output

- Decision Tree Visualization  
- Confusion Matrices and Classification Reports for all models
- Accuracy scores for:
  - Decision Tree
  - Random Forest
  - Random Forest with SMOTE

## 📌 Future Work

- Hyperparameter tuning for improved model performance
- Feature importance analysis
- Deploy model as a web app using Flask or Streamlit

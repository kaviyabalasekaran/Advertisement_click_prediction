# Advertisement Click Prediction

## 📌 Project Description
Advertisement Click Prediction is a **machine learning classification project** aimed at predicting whether a user will click on an advertisement based on their demographic and behavioral data. The dataset used for this project is sourced from Kaggle and contains features such as age, gender, estimated salary, and purchase history.

## 📂 Table of Contents
1. Introduction
2. Dataset Description
3. Goal of the Project
4. Data Preprocessing
5. Model Selection & Implementation
6. Model Comparison & Evaluation
7. Conclusion

## 📊 Dataset
- **Source:** [Kaggle - Social Network Ads Dataset](https://www.kaggle.com/datasets/akram24/social-network-ads)
- **Features:**
  - `Age`
  - `Gender`
  - `Estimated Salary`
  - `Purchased` (Target Variable)

## 🎯 Goal
The primary objective of this project is to build a **predictive model** that accurately classifies whether a user will click on an advertisement or not.

## 🛠️ Technologies & Libraries Used
- **Python**
- **Pandas, NumPy** (Data Handling)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning Models)
- **XGBoost** (Boosting Algorithm)

## 🏗️ Machine Learning Models Used
We implemented and compared the following classification algorithms:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Naive Bayes Classifier**
5. **K-Nearest Neighbors (KNN)**
6. **Support Vector Machine (SVM)**
7. **Gradient Boosting Classifier**
8. **AdaBoost Classifier**
9. **Multi-Layer Perceptron (MLP) Classifier**
10. **XGBoost Classifier**

## 🔬 Model Comparison
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 87% |
| Decision Tree | 86% |
| Random Forest | 92% |
| Naive Bayes | 88% |
| KNN | 93% |
| SVM | 92% |
| Gradient Boosting | 89% |
| AdaBoost | 86% |
| MLP Classifier | 93% |
| XGBoost | 89% |

### **🏆 Best Performing Models:**
- **K-Nearest Neighbors (KNN) and Multi-Layer Perceptron (MLP)** achieved the highest accuracy.
- **Random Forest and SVM** also performed well.

## 📌 Conclusion
This project successfully implemented multiple machine learning algorithms for predicting advertisement clicks. The best models for this dataset were **KNN, MLP, and Random Forest**, achieving high accuracy.

### 🚀 Future Improvements
- Feature Engineering to extract more meaningful insights.
- Hyperparameter tuning for further model optimization.
- Deploying the model using Flask or FastAPI.

## 🔗 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/advertisement-click-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python advertisement_click_prediction.py
   ```

## 📜 License
This project is licensed under the MIT License.


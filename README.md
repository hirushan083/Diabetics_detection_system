🩺 Diabetes Prediction System using Machine Learning

📌 Overview

This project is a machine learning–based diabetes prediction system that predicts the likelihood of diabetes using clinical and lifestyle data. Multiple ML models were trained and evaluated, including Logistic Regression, Random Forest, Support Vector Machine (SVM), and a Neural Network (MLP).
The final trained model is deployed using an interactive Streamlit web application for real-time prediction.

🚀 Features

* Data preprocessing and feature engineering
* Feature scaling using StandardScaler
* Multiple ML models:
  
         Logistic Regression
         Random Forest
         Decision tree
         SVM
         KNN
         Multi-Layer Perceptron (Neural Network)

* Model evaluation using:

        Classification Report
        Confusion Matrix
        ROC Curve & AUC

* Hyperparameter tuning using GridSearchCV
* Deployment-ready Streamlit web application
* Model persistence using .pkl files

📂 Project Structure
* app.py                       # Streamlit application
* diabetes_disease_model.pkl   # Trained ML model
* scaler.pkl                   # Feature scaler
* model-columns.pkl            # Model input feature order
* requirements.txt             # Required libraries
* README.md                    # Project documentation

📊 Dataset

The dataset contains patient health and lifestyle information, including:

* Gender
* Age
* Hypertension
* Heart Disease
* Smoking History
* BMI
* HbA1c Level
* Blood Glucose Level

Target variable:

* diabetes (0 = No, 1 = Yes)

🧠 Machine Learning Workflow

* Data Loading & Exploration
* Data Cleaning & Encoding
* Feature Scaling
* Train–Test Split
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* ROC Curve Analysis
* Model Deployment (Streamlit)

📈 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* ROC Curve & AUC Score (especially for SVM)
* ROC–AUC was prioritized due to class imbalance in medical datasets.

🌐 Streamlit Web Application

 The Streamlit interface allows users to:
 
 * Enter patient health details
 * Predict diabetes risk
 * View prediction probability
 * Get instant results

▶️ Run the App Locally7

    pip install -r requirements.txt
    streamlit run app.py

☁️ Deployment

* The application can be deployed using Streamlit Community Cloud by connecting this repository to GitHub.

🛠️ Technologies Used

* Python
* Google Colab
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* Joblib

📌 Key Highlights

* Neural Network (MLP) with hyperparameter tuning
* ROC curve analysis for SVM
* Use of model-columns.pkl to ensure correct feature alignment during inference
* Deployment-ready ML system

🎓 Academic & Professional Use

This project was developed as part of academic learning and is suitable for:

* Machine Learning portfolios
* Internship applications
* Academic presentations
* Interview demonstrations

👤 Author

* Name: Kavindu Hirushan
* Github: hirushan083
  
      

Loan Default Prediction using Neural Networks
Overview

This project focuses on predicting whether a loan applicant will default using machine learning techniques, specifically neural networks. The dataset used is highly imbalanced, making it challenging to accurately detect defaulters. The project emphasizes handling class imbalance, optimizing model performance, and evaluating results using appropriate metrics.

Problem Statement

Loan default prediction is a binary classification problem where the goal is to determine whether a borrower will default on a loan.

Input: Financial and demographic features of applicants
Output:
0 → Non-default
1 → Default

The dataset is highly imbalanced (~92% non-default, ~8% default), which makes traditional accuracy-based evaluation insufficient.

Dataset
~307,000 records
120+ features
Includes:
Financial attributes (income, credit amount, annuity)
Demographic details (age, employment)
Behavioral indicators
Data Preprocessing

The following preprocessing steps were applied:

Handling missing values:
Numerical → median imputation
Categorical → most frequent value
One-hot encoding for categorical variables
Feature scaling using standardization
Train-test split with preserved class distribution
Model Architecture
Baseline Model
Multilayer Perceptron (MLP)
Layers: 128 → 64 → 1
Activation: ReLU (hidden), Sigmoid (output)
Optimizer: Adam
Optimized Model
Layers: 256 → 128 → 64 → 1
Batch Normalization
Dropout (0.3)
Class weighting
Learning rate scheduling
Early stopping
Evaluation Metrics

Due to class imbalance, multiple metrics were used:

Accuracy
Precision
Recall
F1 Score
Results
Baseline Model
Accuracy: ~0.92
Recall: ~0.01
F1 Score: Very low
Optimized Model
Accuracy: ~0.83
Precision: ~0.22
Recall: ~0.45
F1 Score: ~0.30
Threshold Optimization

Different thresholds were evaluated to balance precision and recall:

Threshold	Precision	Recall	F1 Score
0.3	0.11	0.87	0.20
0.4	0.13	0.76	0.23
0.5	0.16	0.64	0.26
0.6	0.18	0.52	0.27

Best threshold: 0.6

Key Learnings
Accuracy is not reliable for imbalanced datasets
Handling class imbalance is critical
Recall is more important in risk prediction
Threshold tuning improves model usability
Multiple metrics should be used for evaluation
Project Structure
loan-default-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── modeling.ipynb
│
├── models/
│   └── optimized_model.h5
│
├── app/
│   └── app.py
│
├── experiment_log.csv
├── README.md
Technologies Used
Python
Pandas, NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib, Seaborn
Future Work
Apply SMOTE or advanced resampling techniques
Use ensemble models (XGBoost, Random Forest)
Improve feature engineering
Deploy model as a web application
Conclusion

The optimized model significantly improves the detection of loan defaulters compared to the baseline model. Although accuracy decreases, the model becomes more practical and effective by improving recall and achieving a better balance between precision and recall.

Author

Sanket Patil
USN: 4AL22AI043
Fraud & Anomaly Detection

 Project Overview

This project analyzes credit card transactions to identify fraudulent and unusual transaction patterns using machine learning and business intelligence tools.

The project combines **Python, Machine Learning, and Power BI** to explore transaction behavior, detect potential fraud, evaluate model performance, and present the findings through an interactive dashboard.

Two machine learning approaches were used:

* **Random Forest Classifier** – a supervised machine learning model used to classify transactions as legitimate or fraudulent.
* **Isolation Forest** – an unsupervised anomaly detection model used to identify unusual transaction patterns.

 Problem Statement

Credit card fraud is a major challenge for financial institutions because fraudulent transactions represent a very small proportion of total transactions, making them difficult to detect.

The objective of this project is to:

* Analyze credit card transaction patterns.
* Identify fraudulent transactions.
* Detect unusual transaction behavior.
* Compare supervised and unsupervised machine learning approaches.
* Evaluate model performance using precision, recall, and F1-score.
* Build an interactive Power BI dashboard to communicate the results.

Dataset

The project uses the **Credit Card Fraud Detection dataset** from Kaggle.

The dataset contains:

* **284,807 transactions**
* **492 fraudulent transactions**
* **31 columns**
* Transaction time
* Transaction amount
* 28 anonymized features (`V1`–`V28`)
* Fraud classification (`Class`)

The `Class` column contains:

* `0` = Legitimate transaction
* `1` = Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing approximately **0.17%** of all transactions.

Tools & Technologies

 Programming & Data Analysis

* Python
* Pandas
* Matplotlib

 Machine Learning

* Scikit-learn
* Random Forest Classifier
* Isolation Forest
* StandardScaler

 Business Intelligence

* Microsoft Power BI
* DAX
* Data visualization
* Interactive dashboard development

 Development Tools

* Visual Studio Code
* GitHub

 Project Workflow

The project followed these major steps:

1. Loaded and inspected the dataset.
2. Checked for missing values and data quality.
3. Analyzed the distribution of fraudulent and legitimate transactions.
4. Analyzed transaction amounts.
5. Split the dataset into training and testing sets.
6. Scaled the features using StandardScaler.
7. Built an Isolation Forest anomaly detection model.
8. Built a Random Forest classification model.
9. Evaluated both models.
10. Compared model performance.
11. Exported test-set predictions.
12. Built an interactive Power BI dashboard.

 Machine Learning Models

1. Isolation Forest

Isolation Forest was used as an unsupervised anomaly detection technique.

The model identified unusual transactions without using the fraud labels during model training.

Results

| Metric    |  Score |
| --------- | -----: |
| Precision |  4.22% |
| Recall    | 84.69% |
| F1 Score  |  8.03% |

The model achieved high recall but relatively low precision. This means it detected many fraudulent transactions but also classified many legitimate transactions as anomalies.

The model identified **1,968 transactions as anomalies** in the test set.

> Note: These anomalies should not automatically be considered fraudulent transactions.

2. Random Forest Classifier

Random Forest was used as a supervised classification model.

The model was trained using the known fraud labels.

Results

| Metric    |  Score |
| --------- | -----: |
| Precision | 90.59% |
| Recall    | 78.57% |
| F1 Score  | 84.15% |

The Random Forest model predicted **85 transactions as fraudulent** in the test set.

Its higher precision and F1-score made it the stronger-performing model for this dataset.


 Model Comparison

| Model            | Precision | Recall | F1 Score |
| ---------------- | --------: | -----: | -------: |
| Isolation Forest |     4.22% | 84.69% |    8.03% |
| Random Forest    |    90.59% | 78.57% |   84.15% |

 Key Observation

Random Forest performed substantially better overall based on precision and F1-score.

Isolation Forest was useful for identifying unusual transaction behavior, particularly when labeled fraud data is unavailable. However, its large number of false positives significantly reduced its precision.



 Power BI Dashboard

The Power BI dashboard provides an interactive view of the fraud detection analysis.

 Dashboard KPIs

* Test Transactions Analyzed — **56,962**
* Actual Fraud Detected — **98**
* Random Forest Predicted Fraud — **85**
* Isolation Forest Predicted Anomalies — **1,968**
* Random Forest Precision — **90.59%**
* Random Forest Recall — **78.57%**
* Random Forest F1 Score — **84.15%**

 Dashboard Visualizations

* Actual Fraud vs Random Forest Predicted Fraud
* Fraud/Anomaly Predictions by Model
* Average Transaction Amount Over Time
* Average Transaction Amount: Fraud vs Legitimate
* Actual Fraud Transactions Over Time
* Fraud Detection Model Comparison
* Fraud Status slicer

The dashboard allows users to explore transaction behavior and compare actual fraud with machine learning predictions.

 Key Findings

* The dataset is highly imbalanced, with fraudulent transactions accounting for approximately **0.17%** of all transactions.
* Fraud detection requires careful evaluation because accuracy alone can be misleading when fraud cases are rare.
* Isolation Forest achieved high recall but generated many false positives.
* Random Forest achieved **90.59% precision** and an **84.15% F1-score**.
* Random Forest provided a better balance between detecting fraud and minimizing false positives.
* Anomaly detection can still be valuable as a complementary approach for identifying unusual transaction behavior.

 


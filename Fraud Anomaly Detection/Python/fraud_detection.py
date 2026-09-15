import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


data = pd.read_csv("Dataset/creditcard.csv")

print(data.head())

print("Dataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

print("\nTransaction class distribution:")
print(data["Class"].value_counts())

print("\nFraud percentage:")
fraud_percentage = data["Class"].mean() * 100
print(f"{fraud_percentage:.3f}%")

print("\nTransaction amount analysis:")

print("Average legitimate transaction:")
print(data[data["Class"] == 0]["Amount"].mean())

print("Average fraudulent transaction:")
print(data[data["Class"] == 1]["Amount"].mean())

print("\nMaximum legitimate transaction:")
print(data[data["Class"] == 0]["Amount"].max())

print("Maximum fraudulent transaction:")
print(data[data["Class"] == 1]["Amount"].max())

print("\nTotal transaction amount:")

print("Total legitimate transaction value:")
print(data[data["Class"] == 0]["Amount"].sum())

print("Total fraudulent transaction value:")
print(data[data["Class"] == 1]["Amount"].sum())

class_counts = data["Class"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(["Legitimate", "Fraudulent"], class_counts.values)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.show()

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        data[data["Class"] == 0]["Amount"],
        data[data["Class"] == 1]["Amount"]
    ],
    tick_labels=["Legitimate", "Fraudulent"]
)

plt.title("Transaction Amounts: Legitimate vs Fraudulent")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount")
plt.show()

fraud_data = data[data["Class"] == 1]

plt.figure(figsize=(10, 5))
plt.hist(fraud_data["Time"], bins=50)

plt.title("Distribution of Fraudulent Transactions Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Fraudulent Transactions")
plt.show()

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("Testing data shape:")
print(X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled)

print("\nIsolation Forest model trained successfully!")

predictions = model.predict(X_test_scaled)

print("\nAnomaly prediction results:")
print(pd.Series(predictions).value_counts())

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": predictions
})

print("\nActual fraud among detected anomalies:")
print(comparison[comparison["Prediction"] == -1]["Actual"].value_counts())

from sklearn.metrics import classification_report, confusion_matrix

predicted_fraud = (predictions == -1).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, predicted_fraud))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_fraud))

random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

rf_predictions = random_forest.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

isolation_precision = precision_score(y_test, predicted_fraud)
isolation_recall = recall_score(y_test, predicted_fraud)
isolation_f1 = f1_score(y_test, predicted_fraud)

rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\nMODEL COMPARISON")
print("----------------")

print("Isolation Forest:")
print(f"Fraud Precision: {isolation_precision:.2%}")
print(f"Fraud Recall: {isolation_recall:.2%}")
print(f"Fraud F1-Score: {isolation_f1:.2%}")

print("\nRandom Forest:")
print(f"Fraud Precision: {rf_precision:.2%}")
print(f"Fraud Recall: {rf_recall:.2%}")
print(f"Fraud F1-Score: {rf_f1:.2%}")

models = ["Isolation Forest", "Random Forest"]

precision_values = [
    isolation_precision * 100,
    rf_precision * 100
]

recall_values = [
    isolation_recall * 100,
    rf_recall * 100
]

f1_values = [
    isolation_f1 * 100,
    rf_f1 * 100
]

x = range(len(models))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    precision_values,
    width=width,
    label="Precision"
)

plt.bar(
    x,
    recall_values,
    width=width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    f1_values,
    width=width,
    label="F1-Score"
)

plt.xticks(list(x), models)
plt.ylabel("Percentage")
plt.title("Fraud Detection Model Comparison")
plt.legend()

plt.show()

results = X_test.copy()

results["Actual_Fraud"] = y_test.values
results["Isolation_Forest_Prediction"] = predicted_fraud
results["Random_Forest_Prediction"] = rf_predictions

results.to_csv(
    "Dataset/fraud_detection_results.csv",
    index=False
)

print("\nFraud detection results saved successfully!")
print("File: Dataset/fraud_detection_results.csv")
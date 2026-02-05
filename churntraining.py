import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/churn.csv")
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features + Target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model (No scaling)
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
pickle.dump(model, open("model/churn_model.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))

print("✅ Model saved successfully!")

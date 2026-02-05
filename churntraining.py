import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv("data/churn.csv")

# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges datatype
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# ============================
# 2. Encode Categorical Data
# ============================
encoders = {}
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================
# 3. Split Features & Target
# ============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 4. Feature Scaling
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# 5. Train Best Model (XGBoost)
# ============================
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# 6. Evaluate Model
# ============================
y_pred = model.predict(X_test)

print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ============================
# 7. Save Model + Tools
# ============================
pickle.dump(model, open("model/churn_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(encoders, open("model/encoders.pkl", "wb"))

print("\n✅ Model + Scaler + Encoders Saved Successfully!")

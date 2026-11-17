#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              RocCurveDisplay)


# In[ ]:


df=pd.read_csv('data.csv')
df.head()


# In[ ]:


df.info


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.drop(columns=['Unnamed: 32'])


# In[ ]:


df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # M=malignant, B=benign



# In[ ]:


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# In[ ]:


# ---- Scale features (helps logistic regression) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# In[ ]:


# ---- Model 1: Logistic Regression ----
log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train_s, y_train)
y_pred_lr = log_reg.predict(X_test_s)
y_proba_lr = log_reg.predict_proba(X_test_s)[:, 1]


# In[ ]:


# ---- Model 2: Random Forest ----
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]



# In[ ]:


def report(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# In[ ]:


results = [
    report("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    report("Random Forest", y_test, y_pred_rf, y_proba_rf),
]
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("results.csv", index=False)



# In[ ]:


# ---- Confusion matrices ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()


# In[ ]:


# ---- ROC curves ----
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_proba_lr, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="Random Forest", ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()


# In[ ]:


# ---- Feature importance (Random Forest) ----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Top 10 Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nDone. Plots saved.")


# In[ ]:


import pickle

# ---- Load the saved model ----
with open("cancer_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_columns = saved["feature_columns"]


# In[ ]:


new_patient = {
    "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46,
    "area_mean": 566.3, "smoothness_mean": 0.09779, "compactness_mean": 0.08129,
    "concavity_mean": 0.06664, "concave points_mean": 0.04781, "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058,
    "area_se": 23.56, "smoothness_se": 0.008462, "compactness_se": 0.0146,
    "concavity_se": 0.02387, "concave points_se": 0.01315, "symmetry_se": 0.0198,
    "fractal_dimension_se": 0.0023,
    "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7,
    "area_worst": 711.2, "smoothness_worst": 0.144, "compactness_worst": 0.1773,
    "concavity_worst": 0.239, "concave points_worst": 0.1288, "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259,
}


# In[ ]:


# Put values in the exact order the model expects
input_row = pd.DataFrame([new_patient])


# In[ ]:


#  Predict 
prediction = model.predict(input_row)[0]          # 0 = Benign, 1 = Malignant
probability = model.predict_proba(input_row)[0]   # [P(Benign), P(Malignant)]

label = "Malignant" if prediction == 1 else "Benign"

print(f"Prediction: {label}")
print(f"Confidence - Benign: {probability[0]:.2%} | Malignant: {probability[1]:.2%}")


# In[ ]:





# In[ ]:





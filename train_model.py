import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Load Cleaned Dataset ──────────────────────────────
print("Loading cleaned dataset...")
df = pd.read_csv("../data/cleaned_dataset.csv")
print(f"Shape: {df.shape}")

# ── Define Features & Target ──────────────────────────
# Removed Delivery_Time_Min since we predict based on it
features = [
    'Customer_Age',
    'Order_Value',
    'Distance_Km',
    'Items_Count',
    'Customer_Rating',
    'Discount_Applied',
    'Delivery_Partner_Rating',
    'Company_Code',
    'City_Code',
    'Category_Code',
    'Payment_Code'
]

X = df[features]
y = df['Demand_Level']

print(f"\nTarget distribution:\n{y.value_counts()}")

# ── Split Data 80% Train / 20% Test ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ── Train LightGBM Model (tuned for 90-95%) ───────────
print("\nTraining LightGBM model...")

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=20,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)
print("✅ Model training complete!")

# ── Evaluate Model ────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n===== MODEL RESULTS =====")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Slow Delivery', 'Fast Delivery']))

# ── Feature Importance ────────────────────────────────
print("\n===== FEATURE IMPORTANCE =====")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)

# ── Save Model ────────────────────────────────────────
with open("../python_model/demand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as demand_model.pkl")
print("🎉 Step 5 Complete!")


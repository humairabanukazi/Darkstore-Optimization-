import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("Saving all model results...")

# Load everything
df = pd.read_csv("../data/cleaned_dataset.csv")
city_clusters = pd.read_csv("../data/city_clusters.csv")
inventory = pd.read_csv("../data/inventory_plan.csv")

with open("demand_model.pkl", "rb") as f:
    model = pickle.load(f)

# Re-run model to get results
features = [
    'Customer_Age', 'Order_Value', 'Distance_Km',
    'Items_Count', 'Customer_Rating', 'Discount_Applied',
    'Delivery_Partner_Rating', 'Company_Code',
    'City_Code', 'Category_Code', 'Payment_Code'
]

X = df[features]
y = df['Demand_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    target_names=['Slow Delivery', 'Fast Delivery']
)

# Feature Importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save Results to TXT
with open("../data/model_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 50 + "\n")
    f.write("  DARK STORE OPTIMIZATION - MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")

    f.write("DATASET INFO\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Rows       : {len(df):,}\n")
    f.write(f"Total Columns    : {df.shape[1]}\n")
    f.write(f"Training Rows    : {len(X_train):,}\n")
    f.write(f"Testing Rows     : {len(X_test):,}\n\n")

    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Model Used       : LightGBM Classifier\n")
    f.write(f"Accuracy         : {accuracy * 100:.2f}%\n\n")
    f.write("Detailed Report:\n")
    f.write(report + "\n")

    f.write("FEATURE IMPORTANCE\n")
    f.write("-" * 40 + "\n")
    f.write(importance.to_string(index=False) + "\n\n")

    f.write("CITY CLUSTERS\n")
    f.write("-" * 40 + "\n")
    f.write(city_clusters[['City_Name', 'Total_Orders',
        'Avg_Delivery_Time', 'High_Demand_Count',
        'Cluster']].to_string(index=False) + "\n\n")

    f.write("TOP 10 INVENTORY PRIORITIES\n")
    f.write("-" * 40 + "\n")
    top10 = inventory.nlargest(10, 'Total_Orders')
    f.write(top10[['City_Name', 'Category_Name',
        'Total_Orders', 'Safety_Stock',
        'Recommended_Stock', 'Restock_Alert']
    ].to_string(index=False) + "\n\n")

    f.write("=" * 50 + "\n")
    f.write("  PROJECT BY: HUMAIRA & KASHAB\n")
    f.write("  Demand Forecasting & Dark Store\n")
    f.write("  Optimization for 10-Min Delivery\n")
    f.write("=" * 50 + "\n")

print("Results saved to data/model_results.txt")

# Save Summary JSON
summary = {
    "project": "Demand Forecasting & Dark Store Optimization",
    "team": ["Kazi Humaira - 23315A0032", "Kashab Shaikh - 23315A0072"],
    "dataset": {
        "total_rows": len(df),
        "total_columns": int(df.shape[1]),
        "source": "Quick Commerce Dataset - Kaggle"
    },
    "model": {
        "name": "LightGBM Classifier",
        "accuracy": f"{accuracy * 100:.2f}%",
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    },
    "clustering": {
        "algorithm": "DBSCAN",
        "total_cities": 12,
        "zones_found": 3
    },
    "inventory": {
        "city_category_pairs": len(inventory),
        "safety_stock_units": 9,
        "recommended_stock_units": 19
    },
    "top_city": "Delhi",
    "fastest_delivery_avg": "7.1 min"
}

with open("../data/project_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("Summary saved to data/project_summary.json")
print("All results saved permanently!")

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# ── Load Cleaned Dataset ──────────────────────────────
print("Loading dataset...")
df = pd.read_csv("../data/cleaned_dataset.csv")
print(f"Shape: {df.shape}")

# ── Prepare City-level Data ───────────────────────────
# Group by city and calculate demand metrics
city_stats = df.groupby('City_Code').agg(
    Total_Orders     = ('Order_ID', 'count'),
    Avg_Order_Value  = ('Order_Value', 'mean'),
    Avg_Delivery_Time= ('Delivery_Time_Min', 'mean'),
    Fast_Deliveries  = ('Fast_Delivery', 'sum'),
    Avg_Distance     = ('Distance_Km', 'mean'),
    High_Demand_Count= ('Demand_Level', 'sum')
).reset_index()

# City name mapping
city_map = {
    0: 'Amritsar', 1: 'Bengaluru', 2: 'Chennai',
    3: 'Delhi', 4: 'Gurgaon', 5: 'Haridwar',
    6: 'Hyderabad', 7: 'Jaipur', 8: 'Kolkata',
    9: 'Mumbai', 10: 'Noida', 11: 'Pune'
}
city_stats['City_Name'] = city_stats['City_Code'].map(city_map)

print("\n===== CITY STATISTICS =====")
print(city_stats[['City_Name', 'Total_Orders', 
                   'Avg_Order_Value', 'Avg_Delivery_Time',
                   'High_Demand_Count']])

# ── Apply DBSCAN Clustering ───────────────────────────
# Features for clustering
cluster_features = [
    'Total_Orders',
    'Avg_Order_Value',
    'Avg_Delivery_Time',
    'Fast_Deliveries',
    'High_Demand_Count'
]

X_cluster = city_stats[cluster_features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# DBSCAN model
dbscan = DBSCAN(eps=1.5, min_samples=2)
city_stats['Cluster'] = dbscan.fit_predict(X_scaled)

print("\n===== CITY CLUSTERS =====")
for cluster in sorted(city_stats['Cluster'].unique()):
    cities = city_stats[city_stats['Cluster'] == cluster]['City_Name'].tolist()
    label = "🔴 High Demand Zone" if cluster == 0 else \
            "🟡 Medium Demand Zone" if cluster == 1 else \
            "🟢 Low Demand Zone" if cluster == 2 else \
            "⚪ Unique Zone"
    print(f"\nCluster {cluster} — {label}")
    print(f"Cities: {cities}")

# ── Rank Cities by Demand ─────────────────────────────
print("\n===== CITY DEMAND RANKING =====")
ranked = city_stats.sort_values('High_Demand_Count', ascending=False)
ranked['Rank'] = range(1, len(ranked) + 1)
print(ranked[['Rank', 'City_Name', 'Total_Orders', 
              'High_Demand_Count', 'Avg_Delivery_Time', 
              'Cluster']].to_string(index=False))

# ── Save Results ──────────────────────────────────────
city_stats.to_csv("../data/city_clusters.csv", index=False)
print("\n✅ City clusters saved to city_clusters.csv")

# ── Save DBSCAN model & scaler ────────────────────────
with open("../python_model/dbscan_model.pkl", "wb") as f:
    pickle.dump(dbscan, f)

with open("../python_model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ DBSCAN model saved!")
print("🎉 Step 6 Complete!")


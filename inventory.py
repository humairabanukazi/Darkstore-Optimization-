import pandas as pd
import numpy as np

# ── Load Datasets ─────────────────────────────────────
print("Loading datasets...")
df = pd.read_csv("../data/cleaned_dataset.csv")
city_clusters = pd.read_csv("../data/city_clusters.csv")

# City name mapping
city_map = {
    0: 'Amritsar', 1: 'Bengaluru', 2: 'Chennai',
    3: 'Delhi', 4: 'Gurgaon', 5: 'Haridwar',
    6: 'Hyderabad', 7: 'Jaipur', 8: 'Kolkata',
    9: 'Mumbai', 10: 'Noida', 11: 'Pune'
}

category_map = {
    0: 'Beverages', 1: 'Dairy', 2: 'Fruits & Vegetables',
    3: 'Groceries', 4: 'Household', 5: 'Personal Care', 6: 'Snacks'
}

df['City_Name']     = df['City_Code'].map(city_map)
df['Category_Name'] = df['Category_Code'].map(category_map)

# ── Step 1: Demand Per City Per Category ──────────────
print("\nCalculating demand per city per category...")
demand = df.groupby(['City_Name', 'Category_Name']).agg(
    Total_Orders   = ('Order_ID', 'count'),
    Avg_Items      = ('Items_Count', 'mean'),
    Avg_Order_Value= ('Order_Value', 'mean'),
    Std_Items      = ('Items_Count', 'std')
).reset_index()

# ── Step 2: Safety Stock Formula ─────────────────────
# Safety Stock = Z * std_demand * sqrt(lead_time)
# Z = 1.65 (95% service level)
# Lead time = 1 day for dark stores
Z         = 1.65
lead_time = 1

demand['Safety_Stock'] = (
    Z * demand['Std_Items'] * np.sqrt(lead_time)
).round(0).astype(int)

# ── Step 3: Recommended Stock Level ──────────────────
# Recommended = Average daily demand + Safety Stock
demand['Recommended_Stock'] = (
    demand['Avg_Items'] + demand['Safety_Stock']
).round(0).astype(int)

# ── Step 4: Restock Alert ─────────────────────────────
# Flag cities that need priority restocking
avg_orders = demand['Total_Orders'].mean()
demand['Restock_Alert'] = demand['Total_Orders'].apply(
    lambda x: '🔴 HIGH PRIORITY' if x > avg_orders * 1.1
    else '🟡 MEDIUM PRIORITY' if x > avg_orders * 0.9
    else '🟢 LOW PRIORITY'
)

# ── Step 5: Print Results ─────────────────────────────
print("\n===== INVENTORY RECOMMENDATION PER CITY =====")
print(demand[['City_Name', 'Category_Name', 'Total_Orders',
              'Safety_Stock', 'Recommended_Stock', 
              'Restock_Alert']].to_string(index=False))

# ── Step 6: Top 5 City-Category Combinations ─────────
print("\n===== TOP 5 HIGH DEMAND COMBINATIONS =====")
top5 = demand.nlargest(5, 'Total_Orders')
print(top5[['City_Name', 'Category_Name', 
            'Total_Orders', 'Recommended_Stock',
            'Restock_Alert']].to_string(index=False))

# ── Step 7: Save Results ──────────────────────────────
demand.to_csv("../data/inventory_plan.csv", index=False)
print("\n✅ Inventory plan saved to inventory_plan.csv")
print("🎉 Step 7 Complete!")


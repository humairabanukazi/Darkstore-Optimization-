import pandas as pd
import numpy as np

# ── Load ORIGINAL Dataset (not cleaned one) ───────────
print("Loading original dataset...")
df = pd.read_csv("../data/quick_commerce_data_modified_cleaned.csv")
print(f"Original shape: {df.shape}")

# ── Feature Engineering ───────────────────────────────
# Demand label — predicting if delivery will be fast or slow
df['Demand_Level'] = (df['Delivery_Time_Min'] <= 15).astype(int)
print(f"\nFast Deliveries (≤15 min): {df['Demand_Level'].sum()}")
print(f"Slow Deliveries (>15 min): {(df['Demand_Level'] == 0).sum()}")

# Delivery speed category
df['Fast_Delivery'] = (df['Delivery_Time_Min'] <= 10).astype(int)
print(f"Super Fast Deliveries (≤10 min): {df['Fast_Delivery'].sum()}")

# ── Encode text columns into numbers ──────────────────
df['Company_Code']  = df['Company'].astype('category').cat.codes
df['City_Code']     = df['City'].astype('category').cat.codes
df['Category_Code'] = df['Product_Category'].astype('category').cat.codes
df['Payment_Code']  = df['Payment_Method'].astype('category').cat.codes

# ── Save mappings ─────────────────────────────────────
company_map  = dict(enumerate(df['Company'].astype('category').cat.categories))
city_map     = dict(enumerate(df['City'].astype('category').cat.categories))
category_map = dict(enumerate(df['Product_Category'].astype('category').cat.categories))

print("\n===== COMPANY MAPPING =====")
print(company_map)
print("\n===== CITY MAPPING =====")
print(city_map)
print("\n===== CATEGORY MAPPING =====")
print(category_map)

# ── Drop original text columns ────────────────────────
df_clean = df.drop(columns=['Company', 'City', 'Product_Category', 'Payment_Method'])

print("\n===== CLEANED DATASET =====")
print(f"Shape: {df_clean.shape}")
print(df_clean.head())
print("\nColumns:", df_clean.columns.tolist())

# ── Save cleaned file ─────────────────────────────────
output_path = "../data/cleaned_dataset.csv"
df_clean.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to {output_path}")


from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# ── Initialize Flask App ──────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow Node.js to communicate with Flask

# ── Load Trained Models ───────────────────────────────
print("Loading models...")
with open("demand_model.pkl", "rb") as f:
    demand_model = pickle.load(f)

print("✅ Models loaded successfully!")

# ── Mappings ──────────────────────────────────────────
city_map = {
    'Amritsar': 0, 'Bengaluru': 1, 'Chennai': 2,
    'Delhi': 3, 'Gurgaon': 4, 'Haridwar': 5,
    'Hyderabad': 6, 'Jaipur': 7, 'Kolkata': 8,
    'Mumbai': 9, 'Noida': 10, 'Pune': 11
}

company_map = {
    'Amazon Now': 0, 'Big Basket': 1, 'Blinkit': 2,
    'Dunzo': 3, 'Flipkart Minutes': 4, 'Jio Mart': 5,
    'Swiggy Instamart': 6, 'Zepto': 7
}

category_map = {
    'Beverages': 0, 'Dairy': 1, 'Fruits & Vegetables': 2,
    'Groceries': 3, 'Household': 4, 'Personal Care': 5,
    'Snacks': 6
}

payment_map = {
    'Cash on Delivery': 0, 'Credit Card': 1,
    'Debit Card': 2, 'UPI': 3, 'Wallet': 4
}

# ── Home Route ────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': '✅ Flask API Running!',
        'message': 'Dark Store Optimization API',
        'endpoints': ['/predict', '/inventory', '/clusters']
    })

# ── Predict Delivery Speed ────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract inputs
        features = [[
            int(data['customer_age']),
            int(data['order_value']),
            float(data['distance_km']),
            int(data['items_count']),
            int(data['customer_rating']),
            int(data['discount_applied']),
            int(data['delivery_partner_rating']),
            company_map.get(data['company'], 0),
            city_map.get(data['city'], 0),
            category_map.get(data['category'], 0),
            payment_map.get(data['payment_method'], 0)
        ]]

        # Make prediction
        prediction = demand_model.predict(features)[0]
        probability = demand_model.predict_proba(features)[0]

        result = 'Fast Delivery ✅' if prediction == 1 else 'Slow Delivery ⚠️'
        confidence = round(max(probability) * 100, 2)

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence}%",
            'fast_delivery': bool(prediction == 1),
            'recommendation': '✅ Stock is optimized for fast delivery!'
                if prediction == 1
                else '⚠️ Consider restocking or reassigning delivery partner!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── Inventory Route ───────────────────────────────────
@app.route('/inventory', methods=['GET'])
def inventory():
    try:
        df = pd.read_csv("../data/inventory_plan.csv")
        city = request.args.get('city', None)
        if city:
            df = df[df['City_Name'] == city]
        result = df[['City_Name', 'Category_Name',
                     'Total_Orders', 'Safety_Stock',
                     'Recommended_Stock', 'Restock_Alert']].to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── Clusters Route ────────────────────────────────────
@app.route('/clusters', methods=['GET'])
def clusters():
    try:
        df = pd.read_csv("../data/city_clusters.csv")
        result = df[['City_Name', 'Total_Orders',
                     'Avg_Delivery_Time', 'High_Demand_Count',
                     'Cluster']].to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── Run Flask ─────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)


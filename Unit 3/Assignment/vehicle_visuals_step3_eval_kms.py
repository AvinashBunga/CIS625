# vehicle_visuals_step3_eval_kms.py
# 🧪 Step 3: Evaluate regression model for Kilometers vs Price

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# Features
X = df[['Kilometer']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Output
print("📏 Evaluation Metrics for Price vs Kilometers Driven")
print("------------------------------------------------------")
print(f"✅ R² Score        : {r2:.4f}")
print(f"📉 RMSE (INR)      : ₹{int(rmse):,}")
print(f"🧮 Intercept       : ₹{int(model.intercept_):,}")
print(f"🧾 Coefficient     : ₹{int(model.coef_[0]):,} per KM")


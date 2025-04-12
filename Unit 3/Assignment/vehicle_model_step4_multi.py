# vehicle_model_step4_multi.py
# 📊 Step 4: Multiple Linear Regression using Car Age & Kilometers

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# Features and Target
X = df[['Car_Age', 'Kilometer']]
y = df['Price']

# Model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Output
print("📊 Multiple Linear Regression Results (Car_Age + Kilometer)")
print("-----------------------------------------------------------")
print(f"✅ R² Score        : {r2:.4f}")
print(f"📉 RMSE (INR)      : ₹{int(rmse):,}")
print(f"🧮 Intercept       : ₹{int(model.intercept_):,}")
print("🧾 Coefficients:")
print(f"   - Car_Age       : ₹{int(model.coef_[0]):,} per year")
print(f"   - Kilometer     : ₹{int(model.coef_[1]):,} per KM")

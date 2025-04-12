# vehicle_visuals_step3_eval.py
# 🧪 Step 3: Regression evaluation metrics for Car Age vs Price

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# Input features
X = df[['Car_Age']]
y = df['Price']

# Linear Regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Output
print("📏 Evaluation Metrics for Price vs Car Age")
print("-------------------------------------------")
print(f"✅ R² Score        : {r2:.4f}")
print(f"📉 RMSE (INR)      : ₹{int(rmse):,}")
print(f"🧮 Intercept       : ₹{int(model.intercept_):,}")
print(f"🧾 Coefficient     : ₹{int(model.coef_[0]):,} per year")

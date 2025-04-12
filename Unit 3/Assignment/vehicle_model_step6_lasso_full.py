# vehicle_model_step6_lasso_full.py
# 🔍 Lasso Regression with All Features - Step 6

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 📂 Load the cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# ✅ Keep only numeric columns
df = df.select_dtypes(include=['number'])

# 🧼 Drop rows with missing values
df = df.dropna()

# 📊 Split features and target
X = df.drop(columns=['Price'])
y = df['Price']

# 📁 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Lasso model
model = Lasso(alpha=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 📋 Display results
print("\n📊 Lasso Regression Results (All Features - Cleaned & Imputed)")
print("--------------------------------------------------------")
print(f"✅ R² Score        : {r2:.4f}")
print(f"📉 RMSE (INR)      : ₹{rmse:,.0f}")
print("🧾 Top Non-Zero Coefficients:\n")

# 🧮 Show only non-zero coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
non_zero_coefs = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False)
print(non_zero_coefs.head(10).to_string(index=False))

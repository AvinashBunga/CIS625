# vehicle_lasso_model.py
# 🧠 Lasso Regression on CarDekho Dataset (Regularized Linear)

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 🚀 Progress buffer
def progress(stage):
    for p in range(0, 101, 25):
        print(f"{stage}... {p}% done")
        time.sleep(0.2)

# Step 1: Load model-ready data
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_vehicle_data.csv'
print("📥 Step 1: Loading Cleaned Data")
progress("Loading CSV")
df = pd.read_csv(file_path)

# Step 2: Prepare features and target
print("\n🔍 Step 2: Preparing Data")
progress("Selecting Features")
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
X = df.drop(columns=['Price'] + non_numeric_cols)
y = df['Price']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit Lasso model
print("\n🔧 Step 3: Fitting Lasso Regression Model")
progress("Training Model")
lasso_model = Lasso(alpha=1000)  # You can tweak alpha to control regularization strength
lasso_model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = lasso_model.predict(X_test)

# Step 6: Evaluation
print("\n📊 Step 4: Evaluating Lasso Model")
progress("Calculating Metrics")
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"✅ Lasso Model Evaluation:\nRMSE: {rmse:,.2f}\nR² Score: {r2:.4f}")

# Step 7: Show non-zero coefficients
print("\n📉 Step 5: Lasso Regression Coefficients (Non-Zero Only)")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso_model.coef_})
non_zero_coef = coef_df[coef_df['Coefficient'] != 0]
print(non_zero_coef.sort_values(by='Coefficient', key=abs, ascending=False).head(10))  # Top 10

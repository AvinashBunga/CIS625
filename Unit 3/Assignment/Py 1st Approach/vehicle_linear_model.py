# vehicle_linear_model.py
# 📈 Classic Linear Regression on CarDekho Dataset

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 🚀 Progress Buffer Function
def progress(stage):
    for p in range(0, 101, 25):
        print(f"{stage}... {p}% done")
        time.sleep(0.2)

# Step 1: Load model-ready dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_vehicle_data.csv'
print("📥 Step 1: Loading Cleaned Data")
progress("Loading CSV")
df = pd.read_csv(file_path)

# Step 2: Split features and target
print("\n🧪 Step 2: Splitting Features & Target")
progress("Splitting Data")

# Drop all non-numeric columns from features
non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
X = df.drop(columns=['Price'] + non_numeric_cols)
y = df['Price']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit Linear Regression model
print("\n🔧 Step 3: Training Linear Regression Model")
progress("Fitting Model")
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\n📊 Step 4: Evaluating Model")
progress("Calculating Metrics")
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model Evaluation:\nRMSE: {rmse:,.2f}\nR² Score: {r2:.4f}")

# Step 7: Show coefficients
print("\n📉 Step 5: Linear Regression Coefficients")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False).head(10))  # Top 10 important features

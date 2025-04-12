import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Step 1: Load filtered dataset
print("📥 Step 1: Loading Filtered Dataset")
data = pd.read_csv('/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv')
print("✅ Data Loaded!")

# Step 2: Define predictors and target
print("\n🧪 Step 2: Preparing for Quantile Regression")
X = data[['Car_Age', 'Kilometer', 'Length', 'Width', 'Height', 'Fuel Tank Capacity']]
y = data['Price']

# Add constant for intercept
X = sm.add_constant(X)

# Step 3: Run quantile regression at 25th percentile
print("\n🔧 Step 3: Running Quantile Regression (τ = 0.25)")
model = sm.QuantReg(y, X)
res = model.fit(q=0.25)
print("✅ Model Fitted!")

# Step 4: Predict values
y_pred = res.predict(X)

# Step 5: Visualization
print("\n📈 Step 4: Visualizing Quantile Predictions")
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5, label='Predicted vs Actual', color='#3366cc')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Line')
plt.xlabel('Actual Price (INR)', fontsize=12, fontname='Arial')
plt.ylabel('Predicted Price (INR)', fontsize=12, fontname='Arial')
plt.title('Figure 6: Quantile Regression – Actual vs Predicted Car Prices (τ = 0.25)', fontsize=14, fontname='Arial')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.gca().set_facecolor('#f2f2f2')
plt.legend()

# Format axes in full INR values
formatter = FuncFormatter(lambda x, _: f'₹{int(x):,}')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Save the figure
plt.tight_layout()
plt.savefig('Figure_6_Quantile_Prediction.png', dpi=300)
plt.show()

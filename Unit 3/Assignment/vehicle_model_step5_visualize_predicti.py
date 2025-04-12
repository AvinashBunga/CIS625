# vehicle_model_step5_visualize_predictions.py
# 📊 Step 5: Visualize predicted vs actual prices

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from matplotlib.ticker import FuncFormatter

# Load the cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# Define INR format for Y-axis
def inr_format(x, pos):
    return f'₹{int(x):,}'

formatter = FuncFormatter(inr_format)

# Features and Target
X = df[['Car_Age', 'Kilometer']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Scatter plot: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y, y_pred, alpha=0.5, color='mediumseagreen')
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

ax.set_title('Figure 5: Actual vs Predicted Car Prices', fontsize=13)
ax.set_xlabel('Actual Price (INR)')
ax.set_ylabel('Predicted Price (INR)')
ax.set_facecolor('#eeeeee')
ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.legend()

plt.tight_layout()
plt.show()

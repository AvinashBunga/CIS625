# vehicle_visuals_step2_kms.py
# 📈 Step 2: Add linear regression line to Price vs Kilometers Driven plot

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

# Format function for INR and KM with commas
def comma_formatter(x, pos):
    return f'{int(x):,}'

inr_format = FuncFormatter(comma_formatter)
km_format = FuncFormatter(comma_formatter)

# Setup data
X = df[['Kilometer']]
y = df['Price']

# Fit linear model
model = LinearRegression()
model.fit(X, y)

# Predict values across KM range
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Kilometer'], df['Price'], alpha=0.5, color='darkorange', label='Actual Data')
ax.plot(x_range, y_pred, color='red', linewidth=2, label='Best-Fit Line')

ax.set_title('Figure 4: Price vs Kilometers Driven with Best-Fit Line', fontsize=13)
ax.set_xlabel('Kilometers Driven')
ax.set_ylabel('Price (INR)')
ax.set_facecolor('#eeeeee')
ax.grid(True, color='gray', linestyle='--', linewidth=0.4, alpha=0.5)
ax.yaxis.set_major_formatter(inr_format)
ax.xaxis.set_major_formatter(km_format)
ax.legend()

plt.tight_layout()
plt.show()

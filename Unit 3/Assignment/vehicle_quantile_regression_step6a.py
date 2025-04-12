import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
#Load dataset
data = pd.read_csv('/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv')
#Prepare variables
X = data[['Car_Age', 'Kilometer']]
y = data['Price']
#Drop any rows with NaN
X = X.dropna()
y = y.loc[X.index]
#Add constant for intercept
X = sm.add_constant(X)
#Set quantile level
quantile_level = 0.25
#Run Quantile Regression
model = sm.QuantReg(y, X)
result = model.fit(q=quantile_level)
print(result.summary())
#Predict values
y_pred = result.predict(X)
#Format function for Indian numbering system
def format_inr(x, pos):
    return f'₹{int(x):,}'
#Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='royalblue', alpha=0.5, label="Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Line')
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.title(f"Figure 6: Quantile Regression – Actual vs Predicted Car Prices (τ = {quantile_level})")
plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_facecolor('#f5f5f5')  # Soft grey background
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_inr))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_inr))
plt.legend()
plt.tight_layout()
#Save and show plot
plt.savefig("Figure_6_Quantile_Regression.png", dpi=300)
plt.show()

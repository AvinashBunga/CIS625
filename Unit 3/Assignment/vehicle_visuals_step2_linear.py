#Add linear regression best-fit line to Price vs Car Age
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
#Load cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)
#Format function for INR
def comma_formatter(x, pos):
    return f'{int(x):,}'
inr_format = FuncFormatter(comma_formatter)
#Setup data
X = df[['Car_Age']]
y = df['Price']
#Fit linear model
model = LinearRegression()
model.fit(X, y)
#Predict line
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)
#Plot with regression line
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Car_Age'], df['Price'], alpha=0.5, color='dodgerblue', label='Actual Data')
ax.plot(x_range, y_pred, color='red', linewidth=2, label='Best-Fit Line')
ax.set_title('Figure 3: Price vs Car Age with Best-Fit Line', fontsize=13)
ax.set_xlabel('Car Age (Years)')
ax.set_ylabel('Price (INR)')
ax.set_facecolor('#eeeeee')
ax.grid(True, color='gray', linestyle='--', linewidth=0.4, alpha=0.5)
ax.yaxis.set_major_formatter(inr_format)
ax.legend()
plt.tight_layout()
plt.show()
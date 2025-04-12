#Lasso - Actual vs Predicted Visualization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import FuncFormatter
import numpy as np
#Load dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)
#Keep only numeric columns and drop missing values
df = df.select_dtypes(include=['number']).dropna()
# Split into X and y
X = df.drop(columns=['Price'])
y = df['Price']
#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Train Lasso
model = Lasso(alpha=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#INR formatting
def inr_format(x, pos):
    return f'₹{int(x):,}'
formatter = FuncFormatter(inr_format)
#Plot actual vs predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.5, color='dodgerblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction')
ax.set_title('Figure 6: Lasso - Actual vs Predicted Car Prices', fontsize=13)
ax.set_xlabel('Actual Price (INR)')
ax.set_ylabel('Predicted Price (INR)')
ax.set_facecolor('#f4f4f4')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.legend()
plt.tight_layout()
plt.show()

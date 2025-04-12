#vehicle_visuals_step1.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#Load cleaned dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
df = pd.read_csv(file_path)

#Function to format big numbers with commas
def comma_formatter(x, pos):
    return f'{int(x):,}'

inr_format = FuncFormatter(comma_formatter)
km_format = FuncFormatter(comma_formatter)

#Background style function
def set_gray_style(ax):
    ax.set_facecolor('#eeeeee')  # Ultra-soft background
    ax.grid(True, color='gray', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(inr_format)

#Plot 1: Price vs Car Age
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(df['Car_Age'], df['Price'], alpha=0.5, color='dodgerblue')
ax1.set_title('Figure 1: Price vs Car Age', fontsize=13)
ax1.set_xlabel('Car Age (Years)')
ax1.set_ylabel('Price (INR)')
set_gray_style(ax1)
plt.tight_layout()
plt.show()

#Plot 2: Price vs Kilometers Driven
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(df['Kilometer'], df['Price'], alpha=0.5, color='darkorange')
ax2.set_title('Figure 2: Price vs Kilometers Driven', fontsize=13)
ax2.set_xlabel('Kilometers Driven')
ax2.set_ylabel('Price (INR)')
set_gray_style(ax2)
ax2.xaxis.set_major_formatter(km_format)
plt.tight_layout()
plt.show()

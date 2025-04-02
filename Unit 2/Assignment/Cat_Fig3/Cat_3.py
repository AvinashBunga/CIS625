import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Sort experience levels for proper order
exp_order = ['Entry (EN)', 'Mid (MI)', 'Senior (SE)', 'Executive (EX)']

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', order=exp_order, palette="Blues")

# Title and labels
plt.title('Salary Distribution by Experience Level', fontsize=16, fontweight='bold')
plt.xlabel('Experience Level', fontsize=13, fontweight='bold')
plt.ylabel('Salary in USD', fontsize=13, fontweight='bold')

# Format y-axis with commas
locs, labels = plt.yticks()
plt.yticks(locs, [f'{int(val):,}' for val in locs])

# Save figure
plt.tight_layout()
plt.savefig("/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salary_by_experience_boxplot.png")
plt.show()

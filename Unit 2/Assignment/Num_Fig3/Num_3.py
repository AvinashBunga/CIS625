import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Filter relevant columns
df = df[['work_year', 'remote_ratio']]

# Map remote_ratio values to categories
def categorize_remote(val):
    if val == 0:
        return 'No Remote (0%)'
    elif val == 50:
        return 'Hybrid (50%)'
    elif val == 100:
        return 'Fully Remote (100%)'
    else:
        return 'Other'

df['remote_category'] = df['remote_ratio'].apply(categorize_remote)

# Create pivot table
heatmap_data = df.pivot_table(index='remote_category', columns='work_year', aggfunc='size', fill_value=0)

# Sort rows by remote category order
order = ['No Remote (0%)', 'Hybrid (50%)', 'Fully Remote (100%)']
heatmap_data = heatmap_data.reindex(order)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5, linecolor='gray')

# Title and labels
plt.title("Figure 7. Remote Work Ratio Trends by Year (Heatmap)", fontsize=14, weight='bold')
plt.xlabel("Year")
plt.ylabel("Remote Work Category")

# Save plot
output_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/figure7_heatmap_remote_work_trends.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.show()

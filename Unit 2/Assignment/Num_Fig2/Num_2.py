import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Group by year and calculate average salary
salary_by_year = df.groupby('work_year')['salary_in_usd'].mean().reset_index()

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#e0e0e0')  # Light gray background
ax.set_facecolor('#e0e0e0')

# Line chart
ax.plot(salary_by_year['work_year'], salary_by_year['salary_in_usd'],
        marker='o', linestyle='-', color='#2a9d8f', linewidth=3)

# Add text labels
for i, row in salary_by_year.iterrows():
    ax.text(row['work_year'], row['salary_in_usd'] + 2000,
            f"${int(row['salary_in_usd']):,}",
            ha='center', fontsize=10, fontweight='bold', color='black')

# Styling
ax.set_title('Average Salary in AI/ML/Data Science (2020–2025)', fontsize=16, fontweight='bold')
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Salary (USD)', fontsize=13, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)

# Format y-axis labels
locs, labels = ax.get_yticks(), ax.get_yticklabels()
ax.set_yticks(locs)
ax.set_yticklabels([f'{int(val):,}' for val in locs], fontsize=11)

# Save the chart
plt.tight_layout()
plt.savefig("/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/avg_salary_by_year_linechart_grey.png",
            facecolor=fig.get_facecolor())
plt.show()

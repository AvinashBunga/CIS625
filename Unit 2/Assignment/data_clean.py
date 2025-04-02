import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Top 10 most common job titles
top_jobs = df['job_title'].value_counts().nlargest(10)

# Set plot style
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f0f0f0')  # light gray background

# Create bar chart
bars = ax.bar(top_jobs.index, top_jobs.values, color='skyblue')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

# Customize the chart
ax.set_title('Top 10 Most Common Job Titles', fontsize=16, fontweight='bold')
ax.set_xlabel('Job Title', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.tight_layout()

# Save the plot
plt.savefig("/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/top_10_job_titles_labeled.png")
plt.show()

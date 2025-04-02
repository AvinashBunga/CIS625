import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Step 2: Get the top 10 most frequent job titles
top_jobs = df['job_title'].value_counts().nlargest(10)

# Step 3: Plotting the bar chart
plt.figure(figsize=(10, 6))
top_jobs.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Common Job Titles', fontsize=16)
plt.xlabel('Job Title', fontsize=12)
plt.ylabel('Number of Records', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Step 4: Save the figure (optional) and show the plot
plt.savefig("/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/top_10_job_titles.png")
plt.show()

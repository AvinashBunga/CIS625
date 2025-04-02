import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = "/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/salaries.csv"
df = pd.read_csv(file_path)

# Prepare counts, labels, and colors
company_size_counts = df['company_size'].value_counts().sort_values(ascending=False)
label_map = {'M': 'Medium (M)', 'L': 'Large (L)', 'S': 'Small (S)'}
labels = [label_map[k] for k in company_size_counts.index]
counts = company_size_counts.values
colors = ['#2a9d8f', '#e76f51', '#264653']
total = sum(counts)
percentages = [count / total * 100 for count in counts]

# 🚀 Final visual widths (tweaked for clarity)
visual_widths = [65, 22, 13]  # M, L, S – increased S a bit more

# Plot
fig, ax = plt.subplots(figsize=(22, 6))
left = 0
bar_height = 2.5
y_offsets = [0, 0.6, -0.6]

for i, (label, pct, color) in enumerate(zip(labels, percentages, colors)):
    ax.barh(y=0, width=visual_widths[i], height=bar_height, left=left, color=color)
    
    ax.text(left + visual_widths[i] / 2, y_offsets[i],
            f"{label} ({pct:.1f}%)",
            ha='center', va='center',
            color='white', fontsize=16, fontweight='bold')
    
    left += visual_widths[i]

# Final polish
ax.set_xlim(0, sum(visual_widths))
ax.axis('off')
plt.title('Company Size Distribution (Adjusted for Readability)', fontsize=22, fontweight='bold')
plt.tight_layout()

# Save final chart
plt.savefig("/Users/avinash/Desktop/CIS/CIS625/Unit 2/Assignment/company_size_stacked_bar_superfinal.png")
plt.show()

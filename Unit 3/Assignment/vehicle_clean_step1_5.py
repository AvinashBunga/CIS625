# vehicle_clean_step1_5.py
# 🧽 Cleans raw v4 dataset by removing outliers based on real-world logic
# 🔧 Output is saved as cleaned_v4_filtered.csv

import pandas as pd

# Load raw v4 dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/car details v4.csv'
df = pd.read_csv(file_path)

# Calculate Car Age
df['Car_Age'] = 2025 - df['Year']

# 💡 Define thresholds
max_price = 5_000_000       # ₹50 Lakhs
max_km = 200_000            # 2 Lakh kilometers
max_age = 25                # Older than 25 years = outlier

# 🧼 Filter the dataset
filtered_df = df[(df['Price'] <= max_price) &
                 (df['Kilometer'] <= max_km) &
                 (df['Car_Age'] <= max_age)]

# 🔁 Reset index
filtered_df.reset_index(drop=True, inplace=True)

# 💾 Save cleaned version
output_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv'
filtered_df.to_csv(output_path, index=False)

# ✅ Done!
print(f"🧼 Cleaned dataset saved to:\n{output_path}")
print(f"✅ New shape: {filtered_df.shape}")

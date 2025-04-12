# vehicle_analysis.py
# 📁 CarDekho Vehicle Dataset - Initial Exploration Script
# ✍️ Written by Avi

import pandas as pd
import time

# 🚀 Custom buffer progress function
def progress(stage):
    for percent in range(0, 101, 20):
        print(f"{stage}... {percent}% done")
        time.sleep(0.3)  # Simulate processing time

# 1. Load the dataset
print("🔄 Step 1: Loading Dataset")
progress("Reading CSV")
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/car details v4.csv'
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!\n")

# 2. Check shape and sample data
print("📊 Step 2: Dataset Shape and Sample")
progress("Analyzing structure")
print("Rows & Columns:", df.shape)
print("\n🔍 First 5 rows:\n", df.head())

# 3. Data info
print("\n📚 Step 3: Dataset Info")
progress("Fetching column data")
print(df.info())

# 4. Check for missing values
print("\n🧼 Step 4: Checking for Missing Values")
progress("Scanning for nulls")
print(df.isnull().sum())

print("\n✅ All initial checks completed! Ready for cleaning & modeling.")

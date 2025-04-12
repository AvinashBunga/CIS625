# vehicle_cleaning.py
# 🚘 CarDekho Data Cleaning & Preprocessing
# ✍️ Script written by Avi (with signature % buffer!)

import pandas as pd
import time

# 🚀 Custom progress buffer
def progress(stage):
    for percent in range(0, 101, 25):
        print(f"{stage}... {percent}% done")
        time.sleep(0.2)

# 📍 File path to the original dataset
file_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/car details v4.csv'

# Step 1: Load dataset
print("🔧 Starting Data Cleaning")
progress("Loading CSV")
df = pd.read_csv(file_path)

# Step 2: Create Car_Age column
progress("Calculating Car Age")
current_year = 2025
df['Car_Age'] = current_year - df['Year']

# Step 3: Drop irrelevant or text-heavy columns (for ML modeling)
progress("Dropping non-numeric columns")
df.drop(['Color', 'Max Torque', 'Engine', 'Max Power'], axis=1, inplace=True)

# Step 4: Drop rows with missing values (clean data only)
progress("Removing rows with missing values")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Step 5: Save a version for storytelling (with Make + Model)
story_df = df.copy()
storytelling_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/storytelling_vehicle_data.csv'
story_df.to_csv(storytelling_path, index=False)
print(f"\n📁 Storytelling dataset saved to:\n{storytelling_path}")

# Step 6: Prepare data for modeling
progress("Encoding categorical columns")
categorical_cols = ['Make', 'Fuel Type', 'Transmission', 'Location', 'Owner', 'Seller Type', 'Drivetrain']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 7: Save model-ready dataset
model_path = '/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_vehicle_data.csv'
df_encoded.to_csv(model_path, index=False)

print("\n✅ Data cleaned successfully!")
print("📐 Final model-ready dataset shape:", df_encoded.shape)
print(f"📁 Model-ready dataset saved to:\n{model_path}")

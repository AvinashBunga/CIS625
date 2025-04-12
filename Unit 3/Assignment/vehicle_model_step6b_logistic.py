import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Cleaned Data
print("📥 Step 1: Loading Cleaned Data")
data = pd.read_csv('/Users/avinash/Desktop/CIS/CIS625/Unit 3/Assignment/Vehicle dataset/cleaned_v4_filtered.csv')
print("✅ Data Loaded!\n")

# Step 2: Create Binary Target Variable (High vs. Low)
print("🔄 Step 2: Creating Binary Target Column")
median_price = data['Price'].median()
data['Price_Class'] = np.where(data['Price'] > median_price, 'High', 'Low')

# Step 3: Select Features and Target
print("📊 Step 3: Selecting Features for Logistic Regression")
features = ['Car_Age', 'Kilometer', 'Fuel Tank Capacity', 'Length', 'Width']
X = data[features]
y = data['Price_Class']

# Drop missing values if any
X = X.dropna()
y = y[X.index]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train Logistic Regression Model
print("🔧 Step 5: Training Logistic Regression Model")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Step 6: Print Coefficients
print("\n🧾 Logistic Regression Coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature:<22}: {coef:.5f}")

# Step 7: Predict and Generate Confusion Matrix
print("\n📈 Step 6: Generating Confusion Matrix...")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=["Low", "High"])

# Plotting the confusion matrix
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
disp.plot(cmap="Blues", values_format='d')
plt.title("Figure 7: Confusion Matrix – Logistic Regression Model (Values = Car Count)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# Optional: Save figure
plt.savefig("Figure_7 ConfusionMatrix_LogisticRegression.png")

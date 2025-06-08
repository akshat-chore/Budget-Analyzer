import pandas as pd

# Load and clean column names
df = pd.read_csv('data/real_data.csv')
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
df['City_Tier'] = df['City_Tier'].str.extract(r'(\d)').astype(int)




# Display first 5 rows
print("First 5 rows:\n", df.head())

# Display column names
print("\nColumn Names:\n", df.columns.tolist())

# Dataset info
print("\nDataset Info:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Display basic statistics
print("\nSummary Statistics:")
print(df.describe())

# Check unique values in categorical columns
print("\nUnique Values:")
print("Occupations:", df['Occupation'].unique())
print("City Tiers:", df['City_Tier'].unique())

# Visualize: Monthly Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Income'], kde=True, color='skyblue')
plt.title('Monthly Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Visualize: Average spending per category

# List of valid spending categories from your dataset
spending_categories = ['Rent', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

# Calculate average spending
avg_spending = df[spending_categories].mean().sort_values()

# Plot average spending
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
avg_spending.plot(kind='barh', color='teal')
plt.title('Average Monthly Spending by Category')
plt.xlabel('Average Amount (INR)')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

plt.show()

# --- FEATURE ENGINEERING ---

# Clean column names again (safe to repeat)
df.columns = df.columns.str.strip()

# List of monthly expense columns (from your dataset)
expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out',
                'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

# 1. Total Expenses
df['Total_Expenses'] = df[expense_cols].sum(axis=1)

# 2. Actual Savings
df['Actual_Savings'] = df['Income'] - df['Total_Expenses']

# 3. Savings Gap (positive = saving less than desired)
df['Savings_Gap'] = df['Desired_Savings'] - df['Actual_Savings']

# 4. Is Saving Enough? (True = saving at least desired amount)
df['Is_Saving_Enough'] = df['Actual_Savings'] >= df['Desired_Savings']

# Show sample
print("\nNew Features Added:")
print(df[['Income', 'Total_Expenses', 'Actual_Savings', 'Desired_Savings', 'Savings_Gap', 'Is_Saving_Enough']].head())

# --- VISUALIZATION OF NEW FEATURES ---

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Savings vs Income
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Income', y='Actual_Savings', hue='Is_Saving_Enough', palette='Set2')
plt.title('Income vs Actual Savings')
plt.xlabel('Monthly Income')
plt.ylabel('Actual Savings')
plt.tight_layout()
plt.show()

# 2. Histogram: Savings Gap
plt.figure(figsize=(8,5))
sns.histplot(df['Savings_Gap'], kde=True, color='orange')
plt.title('Savings Gap Distribution')
plt.xlabel('Savings Gap (Desired - Actual)')
plt.tight_layout()
plt.show()

# 3. Count plot: Who is saving enough?
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Is_Saving_Enough', palette='coolwarm')
plt.title('Are People Meeting Their Savings Goal?')
plt.xlabel('Saving Enough')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# MACHINE LEARNING - Predicting Savings Status
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

features = ['Income', 'Age', 'Dependents', 'City_Tier', 'Rent', 'Loan_Repayment', 'Groceries', 
            'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
            'Healthcare', 'Education', 'Miscellaneous']


target = 'Is_Saving_Enough'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- RANDOM FOREST CLASSIFIER ---

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict using RF
rf_pred = rf_model.predict(X_test)

# Evaluation
print("\n[Random Forest] Accuracy:", accuracy_score(y_test, rf_pred))
print("\n[Random Forest] Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\n[Random Forest] Classification Report:\n", classification_report(y_test, rf_pred))

# --- COMPARE RESULTS ---
lr_acc = accuracy_score(y_test, y_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n🔍 Accuracy Comparison:")
print(f"Logistic Regression: {lr_acc:.2f}")
print(f"Random Forest: {rf_acc:.2f}")

if rf_acc > lr_acc:
    print("✅ Random Forest performed better!")
elif lr_acc > rf_acc:
    print("✅ Logistic Regression performed better!")
else:
    print("⚖️ Both models performed equally.")

# --- XGBOOST CLASSIFIER ---
from xgboost import XGBClassifier

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluation
print("\n[XGBoost] Accuracy:", accuracy_score(y_test, xgb_pred))
print("\n[XGBoost] Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("\n[XGBoost] Classification Report:\n", classification_report(y_test, xgb_pred))

# --- FINAL COMPARISON ---
xgb_acc = accuracy_score(y_test, xgb_pred)

print("\n🏁 Final Accuracy Comparison:")
print(f"Logistic Regression: {accuracy_score(y_test, y_pred):.2f}")
print(f"Random Forest:       {accuracy_score(y_test, rf_pred):.2f}")
print(f"XGBoost:             {xgb_acc:.2f}")

best_model = max([(lr_acc, 'Logistic Regression'), (rf_acc, 'Random Forest'), (xgb_acc, 'XGBoost')])
print(f"\n🔥 Best Performing Model: {best_model[1]} with Accuracy: {best_model[0]:.2f}")

print(f"\n🔥 Best Performing Model: {best_model[1]} with Accuracy: {best_model[0]:.2f}")

# --- CLUSTERING FINANCIAL BEHAVIOR ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("\n--- CLUSTERING FINANCIAL BEHAVIOR ---")

cluster_features = df[['Income', 'Total_Expenses', 'Actual_Savings', 'Savings_Gap']].copy()

# Normalize
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Spending_Cluster'] = kmeans.fit_predict(cluster_scaled)

# Show centers
print("\nCluster Centers:")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                   columns=cluster_features.columns))

# Show counts
print("\nCluster Counts:")
print(df['Spending_Cluster'].value_counts())

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Income'], y=df['Total_Expenses'], hue=df['Spending_Cluster'], palette='Set2')
plt.title('Financial Behavior Clusters')
plt.xlabel('Income')
plt.ylabel('Total Expenses')
plt.tight_layout()
plt.show()



input("Press Enter to exit...")

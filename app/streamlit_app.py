import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Title ---
st.title("💸 Budget & Savings Analyzer (ML-Powered)")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your financial data (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Sample Data")
    st.dataframe(df.head())

    # --- Feature Engineering ---
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out',
                    'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

    df['Total_Expenses'] = df[expense_cols].sum(axis=1)
    df['Actual_Savings'] = df['Income'] - df['Total_Expenses']
    df['Savings_Gap'] = df['Desired_Savings'] - df['Actual_Savings']
    df['Is_Saving_Enough'] = df['Actual_Savings'] >= df['Desired_Savings']
    df['City_Tier'] = df['City_Tier'].str.extract('(\d)').astype(int)

    st.subheader("📈 Financial Insights")
    st.write(df[['Income', 'Total_Expenses', 'Actual_Savings', 'Savings_Gap', 'Is_Saving_Enough']].head())

    # --- Train a Quick Model ---
    features = ['Income', 'Age', 'Dependents', 'City_Tier'] + expense_cols
    X = df[features]
    y = df['Is_Saving_Enough']

    model = RandomForestClassifier()
    model.fit(X, y)
    df['Model_Prediction'] = model.predict(X)

    st.subheader("🤖 ML Prediction: Saving Enough?")
    st.write(df[['Income', 'Actual_Savings', 'Desired_Savings', 'Model_Prediction']].head())

    # --- Clustering ---
    cluster_features = df[['Income', 'Total_Expenses', 'Actual_Savings', 'Savings_Gap']]
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Spending_Cluster'] = kmeans.fit_predict(cluster_scaled)

    st.subheader("🧠 Spending Cluster Analysis")
    st.write(df['Spending_Cluster'].value_counts())

    # --- Cluster Plot ---
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Income'], y=df['Total_Expenses'], hue=df['Spending_Cluster'], palette='Set2', ax=ax)
    plt.title("Clusters by Income & Spending")
    st.pyplot(fig)

    # --- Download Enhanced Data ---
    st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="enhanced_financial_data.csv")

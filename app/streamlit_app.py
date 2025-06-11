import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# --- Page Config ---
st.set_page_config(page_title="Budget & Savings Analyzer", layout="wide")
st.title("💸 ML-Powered Budget & Savings Analyzer")

# --- Cached Functions ---
@st.cache_data(ttl=300)
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_rf_model(X, y):
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_kmeans(data, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(scaled_data)

# --- Sample Dataset Download ---
with st.expander("📁 Don't have data? Download a sample dataset to try the app"):
    try:
        sample_df = load_data("data/budget_data.csv")
        st.download_button(
            label="📥 Download Sample Dataset",
            data=sample_df.to_csv(index=False),
            file_name="sample_budget_data.csv",
            mime="text/csv"
        )
    except Exception:
        st.error("❌ Sample dataset not found. Ensure it exists at 'data/budget_data.csv'.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your financial data (.csv)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📊 Sample Data")
    st.dataframe(df.head(10))  # Display fewer rows for faster UI rendering

    # --- Feature Engineering ---
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out',
                    'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

    df['Total_Expenses'] = df[expense_cols].sum(axis=1)
    df['Actual_Savings'] = df['Income'] - df['Total_Expenses']
    df['Savings_Gap'] = df['Desired_Savings'] - df['Actual_Savings']
    df['Is_Saving_Enough'] = df['Actual_Savings'] >= df['Desired_Savings']
    df['City_Tier'] = df['City_Tier'].str.extract('(\d)').astype(int)

    st.subheader("📈 Financial Insights")
    st.dataframe(df[['Income', 'Total_Expenses', 'Actual_Savings', 'Savings_Gap', 'Is_Saving_Enough']].head(10))

    # --- ML Prediction ---
    st.subheader("🤖 ML Prediction: Saving Enough?")
    features = ['Income', 'Age', 'Dependents', 'City_Tier'] + expense_cols
    X = df[features]
    y = df['Is_Saving_Enough']
    rf_model = train_rf_model(X, y)
    df['Model_Prediction'] = rf_model.predict(X)
    st.dataframe(df[['Income', 'Actual_Savings', 'Desired_Savings', 'Model_Prediction']].head(10))

    # --- Clustering ---
    st.subheader("🧠 Spending Cluster Analysis")
    cluster_features = df[['Income', 'Total_Expenses', 'Actual_Savings', 'Savings_Gap']]
    df['Spending_Cluster'] = train_kmeans(cluster_features)
    st.write(df['Spending_Cluster'].value_counts())

    # --- Cluster Plot ---
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Income'], y=df['Total_Expenses'], hue=df['Spending_Cluster'], palette='Set2', ax=ax)
    plt.title("Clusters by Income & Spending")
    st.pyplot(fig)

    # --- Download Results ---
    st.download_button("📥 Download Enhanced Data as CSV", df.to_csv(index=False), file_name="enhanced_financial_data.csv")
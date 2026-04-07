import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Shopping Behavior Dashboard", layout="wide")
st.title("🛒 Consumer Shopping Behavior Analysis Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Consumer_Shopping_Trends_2026.csv")
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()
df = df_raw.copy()

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio("Navigation", [
    "📊 Dataset Overview",
    "📈 Statistical Analysis",
    "🧹 Data Cleaning",
    "⚙ Feature Engineering",
    "📊 EDA & Visualizations",
    "🤖 Model & Evaluation"
])

# =========================
# DATASET OVERVIEW
# =========================
if menu == "📊 Dataset Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col1.metric("Columns", df.shape[1])
    col2.write(df.dtypes)

# =========================
# STATISTICAL ANALYSIS
# =========================
elif menu == "📈 Statistical Analysis":
    st.header("Statistical Analysis")

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Mean / Median / Mode")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Mean")
        st.write(df.mean(numeric_only=True))

    with col2:
        st.write("### Median")
        st.write(df.median(numeric_only=True))

    with col3:
        st.write("### Mode")
        st.write(df.mode().iloc[0])

    st.subheader("Correlation Matrix")
    st.write(df.select_dtypes(include=['int64','float64']).corr())

# =========================
# DATA CLEANING
# =========================
elif menu == "🧹 Data Cleaning":
    st.header("Data Cleaning & Preprocessing")

    # =========================
    # BEFORE CLEANING
    # =========================
    st.subheader("Before Cleaning")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Missing Values")
        st.write(df_raw.isnull().sum())

    with col2:
        st.write("Duplicate Rows")
        st.write(df_raw.duplicated().sum())

    # =========================
    # CLEANING STEPS
    # =========================
    df_clean = df_raw.copy()

    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)

    # Strip column names
    df_clean.columns = df_clean.columns.str.strip()

    # Fill missing numeric with median
    for col in df_clean.select_dtypes(include=['int64','float64']).columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Fill categorical with mode
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # =========================
    # AFTER CLEANING
    # =========================
    st.subheader("After Cleaning")

    col3, col4 = st.columns(2)

    with col3:
        st.write("Missing Values")
        st.write(df_clean.isnull().sum())

    with col4:
        st.write("Duplicates Remaining")
        st.write(df_clean.duplicated().sum())

    # =========================
    # OUTLIER DETECTION 🔥
    # =========================
    st.subheader("Outlier Detection (Boxplots)")

    numeric_cols = df_clean.select_dtypes(include=['int64','float64']).columns

    selected_col = st.selectbox("Select column to check outliers", numeric_cols)

    fig, ax = plt.subplots()
    sns.boxplot(y=df_clean[selected_col], ax=ax)
    st.pyplot(fig)

    # =========================
    # CATEGORICAL DISTRIBUTION
    # =========================
    st.subheader("Categorical Value Distribution")

    cat_cols = df_clean.select_dtypes(include=['object']).columns

    selected_cat = st.selectbox("Select categorical column", cat_cols)

    st.write(df_clean[selected_cat].value_counts())

    fig, ax = plt.subplots()
    df_clean[selected_cat].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # =========================
    # DATA TYPE INFO
    # =========================
    st.subheader("Data Types After Cleaning")
    st.write(df_clean.dtypes)

    # =========================
    # FINAL CLEAN DATA PREVIEW
    # =========================
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df_clean.head())

    # =========================
    # INSIGHTS
    # =========================
    st.success("""
    ✔ Missing values handled using median/mode  
    ✔ Duplicate rows removed  
    ✔ Data standardized  
    ✔ Outliers visualized for further treatment  
    """)
# =========================
# FEATURE ENGINEERING (RESTORED PROPERLY)
# =========================
elif menu == "⚙ Feature Engineering":
    st.header("Feature Engineering")

    df_fe = df.copy()

    # Original features
    df_fe['total_spend'] = df_fe['avg_online_spend'] + df_fe['avg_store_spend']
    df_fe['online_ratio'] = df_fe['monthly_online_orders'] / (
        df_fe['monthly_online_orders'] + df_fe['monthly_store_visits'] + 1
    )

    st.subheader("New Features Created")
    st.write("""
    - total_spend → total customer spending
    - online_ratio → online vs offline behavior
    """)

    st.subheader("Dataset After Feature Engineering")
    st.dataframe(df_fe.head())

    st.info("""
    Insights:
    - High total_spend → premium customer
    - High online_ratio → prefers online shopping
    """)

# =========================
# EDA (FULL RESTORED VERSION 🔥)
# =========================
elif menu == "📊 EDA & Visualizations":
    st.header("Interactive Exploratory Data Analysis")

    df_eda = df.copy()

    numeric_cols = df_eda.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df_eda.select_dtypes(include=['object']).columns.tolist()

    # =========================
    # FILTER SECTION
    # =========================
    st.sidebar.subheader("🔍 Filters")

    selected_cat_filter = None
    if len(cat_cols) > 0:
        filter_col = st.sidebar.selectbox("Filter by category", ["None"] + cat_cols)

        if filter_col != "None":
            unique_vals = df_eda[filter_col].unique()
            selected_val = st.sidebar.selectbox("Select value", unique_vals)
            df_eda = df_eda[df_eda[filter_col] == selected_val]

    # =========================
    # PLOT SELECTION
    # =========================
    st.subheader("Choose Visualization")

    plot_type = st.selectbox("Select Plot Type", [
        "Correlation Heatmap",
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Bar Chart",
        "Pie Chart"
    ])

    # =========================
    # HEATMAP
    # =========================
    if plot_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df_eda[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # =========================
    # HISTOGRAM
    # =========================
    elif plot_type == "Histogram":
        col = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df_eda[col], kde=True, ax=ax)
        st.pyplot(fig)

    # =========================
    # BOX PLOT
    # =========================
    elif plot_type == "Box Plot":
        x_col = st.selectbox("X (categorical)", cat_cols)
        y_col = st.selectbox("Y (numeric)", numeric_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=x_col, y=y_col, data=df_eda, ax=ax)
        st.pyplot(fig)

    # =========================
    # SCATTER
    # =========================
    elif plot_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)

        hue_col = st.selectbox("Color by", ["None"] + cat_cols)

        fig, ax = plt.subplots()

        if hue_col == "None":
            sns.scatterplot(x=x_col, y=y_col, data=df_eda, ax=ax)
        else:
            sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df_eda, ax=ax)

        st.pyplot(fig)

    # =========================
    # BAR CHART
    # =========================
    elif plot_type == "Bar Chart":
        col = st.selectbox("Select categorical column", cat_cols)

        fig, ax = plt.subplots()
        df_eda[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # =========================
    # PIE CHART
    # =========================
    elif plot_type == "Pie Chart":
        col = st.selectbox("Select categorical column", cat_cols)

        fig, ax = plt.subplots()
        df_eda[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    # =========================
    # QUICK INSIGHTS
    # =========================
    st.subheader("Quick Insights")

    if len(numeric_cols) > 0:
        col = st.selectbox("Select column for stats", numeric_cols)
        st.write("Mean:", df_eda[col].mean())
        st.write("Median:", df_eda[col].median())
        st.write("Std Dev:", df_eda[col].std())

# =========================
# MODEL & EVALUATION
# =========================
elif menu == "🤖 Model & Evaluation":

    df_model = df.copy()

    # Feature Engineering
    df_model['total_spend'] = df_model['avg_online_spend'] + df_model['avg_store_spend']
    df_model['online_ratio'] = df_model['monthly_online_orders'] / (
        df_model['monthly_online_orders'] + df_model['monthly_store_visits'] + 1
    )

    y_cat = df_model['shopping_preference'].astype('category')
    label_map = dict(enumerate(y_cat.cat.categories))
    y = y_cat.cat.codes

    X = df_model.drop('shopping_preference', axis=1)
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.success(f"Accuracy: {round(accuracy_score(y_test, y_pred)*100,2)}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Report
    st.text(classification_report(y_test, y_pred))

    # Prediction
    st.subheader("Predict Customer")

    age = st.slider("Age", 18, 80, 30)
    income = st.slider("Income", 10000, 200000, 50000)
    internet = st.slider("Internet Hours", 0.5, 15.0, 5.0)
    tech = st.slider("Tech Score", 1, 10, 5)
    impulse = st.slider("Impulse Score", 1, 10, 5)
    online_orders = st.slider("Online Orders", 0, 50, 5)
    store_visits = st.slider("Store Visits", 0, 50, 5)
    avg_online_spend = st.slider("Online Spend", 0, 50000, 5000)
    avg_store_spend = st.slider("Store Spend", 0, 50000, 5000)

    gender = st.selectbox("Gender", ["Male","Female","Other"])
    city = st.selectbox("City Tier", ["Tier 1","Tier 2","Tier 3"])

    input_df = pd.DataFrame({
        'age':[age],
        'monthly_income':[income],
        'daily_internet_hours':[internet],
        'tech_savvy_score':[tech],
        'impulse_buying_score':[impulse],
        'monthly_online_orders':[online_orders],
        'monthly_store_visits':[store_visits],
        'avg_online_spend':[avg_online_spend],
        'avg_store_spend':[avg_store_spend],
        'gender':[gender],
        'city_tier':[city]
    })

    input_df['total_spend'] = input_df['avg_online_spend'] + input_df['avg_store_spend']
    input_df['online_ratio'] = input_df['monthly_online_orders'] / (
        input_df['monthly_online_orders'] + input_df['monthly_store_visits'] + 1
    )

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)

    st.success(f"Predicted: {label_map[pred[0]]}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Title of the app
st.title("Data Analysis and Machine Learning")
st.sidebar.title("Navigation")

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('Amazon_Sale_Report.csv')  # Replace with your dataset's path
    return data

# Function to clean data
def clean_data(data):
    # Example cleaning steps
    data.dropna(inplace=True)  # Remove missing values
    return data

# Main app workflow
data = load_data()

# Sidebar options
options = st.sidebar.radio("Choose an option:", 
                            ["Show Raw Data", "Clean Data", "Explore Data", "Visualize Data", "Apply ML Algorithm"])

if options == "Show Raw Data":
    st.header("Raw Data")
    st.write(data)

elif options == "Clean Data":
    st.header("Cleaned Data")
    cleaned_data = clean_data(data)
    st.write(cleaned_data)
    
    # Download cleaned data
    csv = cleaned_data.to_csv(index=False)
    st.download_button(
        label="Download Cleaned Data",
        data=csv,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )

elif options == "Explore Data":
    st.header("Data Exploration")
    st.write("Summary statistics:")
    st.write(data.describe())

    # Column-wise statistics
    column = st.selectbox("Select a column to view details", data.columns)
    st.write(f"Unique values in {column}: {data[column].nunique()}")
    st.write(f"Value counts in {column}:")
    st.write(data[column].value_counts())

elif options == "Visualize Data":
    st.header("Visualize Data")
    
    chart_type = st.selectbox("Select chart type", ["Line Plot", "Bar Plot", "Scatter Plot"])
    x_axis = st.selectbox("Select X-axis", data.columns)
    y_axis = st.selectbox("Select Y-axis", data.columns)
    
    if chart_type == "Line Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Bar Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Scatter Plot":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

elif options == "Apply ML Algorithm":
    st.header("Machine Learning Models")

    # Select features and target
    st.subheader("Feature Selection")
    features = st.multiselect("Select Feature Columns:", data.columns)
    target = st.selectbox("Select Target Column:", data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Select ML algorithm
        st.subheader("Choose an Algorithm")
        algorithm = st.selectbox(
            "Algorithm",
            [
                "Linear Regression",
                "Logistic Regression",
                "K-Means Clustering",
                "Naive Bayes",
                "Support Vector Machine (SVM)",
                "Random Forest",
                "Decision Tree",
            ],
        )

        model = None
        if algorithm == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

        elif algorithm == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "K-Means Clustering":
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=n_clusters)
            model.fit(X)
            data['Cluster'] = model.labels_
            st.write("Clustered Data:")
            st.write(data)

        elif algorithm == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "Support Vector Machine (SVM)":
            model = SVC()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "Random Forest":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        st.success(f"{algorithm} applied successfully!")
    else:
        st.warning("Please select features and target to proceed.")

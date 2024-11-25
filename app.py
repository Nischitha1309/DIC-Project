import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("E-commerce and Retail Sales Data Analysis")
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
                            ["Show Raw Data", "Clean Data", "Explore Data", "Visualize Data"])

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

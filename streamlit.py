import streamlit as st
import pandas as pd
import numpy as np
# Uncomment if `matplotlib` and `seaborn` are installed
# import matplotlib.pyplot as plt
# import seaborn as sns

# Title of the app
st.title("Data Analysis and Visualization App")
st.sidebar.title("Navigation")

# Upload dataset
st.sidebar.header("Upload your dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Load dataset
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Function to clean data
def clean_data(data):
    # Example cleaning steps
    data = data.dropna()  # Remove missing values
    return data

# Main workflow
if uploaded_file:
    data = load_data(uploaded_file)

    # Sidebar options
    options = st.sidebar.radio(
        "Choose an option:",
        ["Show Raw Data", "Clean Data", "Explore Data", "Visualize Data"]
    )

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
            file_name="cleaned_data.csv",
            mime="text/csv"
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
        
        # Select chart type and axes
        chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Scatter Chart"])
        x_axis = st.selectbox("Select X-axis", data.columns)
        y_axis = st.selectbox("Select Y-axis", data.columns)
        
        # Streamlit's built-in charts (Uncomment matplotlib-based code if needed)
        if chart_type == "Line Chart":
            st.line_chart(data[[x_axis, y_axis]].set_index(x_axis))
        elif chart_type == "Bar Chart":
            st.bar_chart(data[[x_axis, y_axis]].set_index(x_axis))
        elif chart_type == "Scatter Chart":
            st.write("Scatter plot is not supported by Streamlit natively. Please enable seaborn.")
            
            # Uncomment the following code if `matplotlib` and `seaborn` are installed
            # fig, ax = plt.subplots(figsize=(10, 6))
            # sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
            # st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")

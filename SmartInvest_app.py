
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Title and description
st.title("SmartInvest: MOORA-Based Stock Selection for Educational Innovation")
st.markdown(""" 
This app evaluates and ranks stocks based on multiple criteria using the **MOORA (Multi-Objective Optimization on the Basis of Ratio Analysis)** method. 
Upload your stock dataset, specify weights for each criterion, and the system will compute rankings based on the MOORA method.
""") 

# File uploader for decision matrix (stock data)
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example fallback dataset
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load the data
df = None
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

# Display the uploaded data or example
st.subheader("Stock Data")
st.dataframe(df)

# Extract stock names and criteria
stocks = df.iloc[:, 0]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Input weights for each criterion
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1/len(criteria), step=0.01)
    weights.append(weight)

# Ensure weights sum to 1
if sum(weights) != 1:
    st.warning("Weights must sum to 1! Please adjust the weights.")

# Normalize the data using vector normalization
st.subheader("Step 1: Normalize the Data")
normalized = data.copy()
for i, col in enumerate(criteria):
    norm = data[col] / np.sqrt((data[col]**2).sum())
    normalized[col] = norm
st.dataframe(normalized)

# Weighted Normalized Matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted = normalized.copy()
for i, col in enumerate(criteria):
    weighted[col] = weighted[col] * weights[i]
st.dataframe(weighted)

# Calculate MOORA Performance Index (PIS)
st.subheader("Step 3: MOORA Performance Index (PIS)")
positive_ideal_solution = weighted.max()
negative_ideal_solution = weighted.min()

# Euclidean Distance from PIS and NIS
distance_pis = np.sqrt(((weighted - positive_ideal_solution)**2).sum(axis=1))
distance_nis = np.sqrt(((weighted - negative_ideal_solution)**2).sum(axis=1))

# Calculate the relative closeness
relative_closeness = distance_nis / (distance_pis + distance_nis)

# Calculate the final rankings based on the relative closeness
st.subheader("Step 4: Final Rankings")
ranking = pd.DataFrame({
    'Stock': stocks,
    'Distance from PIS': distance_pis,
    'Distance from NIS': distance_nis,
    'Relative Closeness': relative_closeness
})
ranking = ranking.sort_values(by="Relative Closeness", ascending=False).reset_index(drop=True)

# Highlight the top-ranked stock
def highlight_top(row):
    return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

st.dataframe(ranking.style.apply(highlight_top, axis=1))

# Download Results as CSV
st.subheader("Download Result")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(ranking)
st.download_button("Download Results as CSV", csv, "smartinvest_results.csv", "text/csv")

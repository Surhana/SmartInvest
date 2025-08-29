import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("SmartInvest: MOORA-Based Stock Selection System")
st.markdown("""
This app evaluates and ranks stocks based on multiple criteria using the **MOORA** method.
Upload your stock dataset, assign weights to each criterion, choose benefit/cost criteria,
and view rankings with tables and a bar chart.
""")

# File uploader
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example data
def load_example():
    data = {
        'Alternative': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

# Ensure first column is Alternative
df = df.rename(columns={df.columns[0]: "Alternative"})

st.subheader(" Data")
st.dataframe(df)

alternatives = df["Alternative"]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# ---------------- STEP 0: Input weights & Select criteria ----------------
st.subheader("Input Weights and Select Criteria")
weights = [st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, 
                           value=1/len(criteria), step=0.01) for col in criteria]

weight_sum = round(sum(weights),4)
st.write(f"**Current weight sum:** {weight_sum}")

# Select benefit and cost criteria below weight inputs
benefit_criteria = st.multiselect("Select Benefit Criteria", criteria.tolist())
cost_criteria = st.multiselect("Select Cost Criteria", [c for c in criteria if c not in benefit_criteria])

if weight_sum != 1:
    st.warning("Weights must sum to 1 to proceed.")
elif not benefit_criteria and not cost_criteria:
    st.warning("Please select based on their criterion.")
else:
    # ---------------- STEP 1: Normalize ----------------
    st.subheader("Step 1: Normalize the Data")
    normalized_matrix = data.copy()
    for i, col in enumerate(criteria):
        normalized_matrix[col] = data[col] / np.sqrt((data[col]**2).sum())
    normalized_df = pd.concat([alternatives, normalized_matrix], axis=1)
    st.dataframe(normalized_df)

    # ---------------- STEP 2: Weighted Normalized Matrix ----------------
    st.subheader("Step 2: Weighted Normalized Matrix")
    weighted_matrix = normalized_matrix.copy()
    for i, col in enumerate(criteria):
        weighted_matrix[col] = normalized_matrix[col] * weights[i]
    weighted_df = pd.concat([alternatives, weighted_matrix], axis=1)
    st.dataframe(weighted_df)

    # ---------------- STEP 3: Calculate Performance Score ----------------
    st.subheader("Step 3: Calculate Performance Score")

    # Compute MOORA (Performance Score)
    benefit_data = weighted_matrix[benefit_criteria] if benefit_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))
    cost_data = weighted_matrix[cost_criteria] if cost_criteria else pd.DataFrame(np.zeros((len(alternatives),0)))

    performance_score = benefit_data.sum(axis=1) - cost_data.sum(axis=1)

    perf_df = pd.DataFrame({
        "Alternative": alternatives,
        "Performance Score": performance_score.round(4)
    })
    st.dataframe(perf_df)

    # ---------------- STEP 4: Final Rankings ----------------
    st.subheader("Step 4: Final Rankings")
    ranking = perf_df.sort_values('Performance Score', ascending=False).reset_index(drop=True)
    ranking['Rank'] = range(1, len(ranking) + 1)

    # Format Performance Score to 4 decimals in ranking table
    ranking['Performance Score'] = ranking['Performance Score'].map('{:.4f}'.format)

    # Highlight the top-ranked alternative in green
    def highlight_top(row):
        return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

    st.dataframe(ranking.style.apply(highlight_top, axis=1))

    # Announce the best alternative
    best_alt = ranking.loc[0, 'Alternative']
    st.success(f"🏆 **The Best Alternative is:** {best_alt} ")

    # ---------------- STEP 5: Vertical Bar Chart ----------------
    st.subheader("Ranking the Chart")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(ranking['Alternative'], ranking['Performance Score'].astype(float), color='skyblue')
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ---------------- DOWNLOAD CSV ----------------
    st.subheader("Download Result")
    csv = ranking.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", csv, "edurank_results.csv", "text/csv")

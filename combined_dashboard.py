
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    # Find the dataset anywhere inside the project folder
    files = list(PROJECT_DIR.rglob("cleaned_telco_churn.csv"))

    if not files:
        files = list(
            PROJECT_DIR.rglob("Telco_Customer_Churn_Dataset.csv")
        )

    if not files:
        return None

    data = pd.read_csv(files[0])
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return data


df = load_data()

st.title("📊 Customer Segmentation & Churn Analysis")
st.caption("Saiket Systems | Business Analysis Internship")

if df is None:
    st.error("Dataset not found inside your project folder.")
    st.write(
        "Please check that Telco_Customer_Churn_Dataset.csv "
        "is inside one of the Task folders."
    )
    st.stop()

st.success("Dataset loaded successfully!")

# Find columns even if the dataset uses different column names
def find_column(options):
    for column in options:
        if column in df.columns:
            return column
    return None


tenure_col = find_column(["tenure", "tenure_months"])
charges_col = find_column(
    ["monthly_charges", "monthlycharges"]
)
churn_col = find_column(["churn"])
gender_col = find_column(["gender"])
contract_col = find_column(["contract", "contract_type"])
payment_col = find_column(
    ["payment_method", "paymentmethod"]
)
senior_col = find_column(["senior_citizen", "seniorcitizen"])

# Convert numeric columns when available
for col in [tenure_col, charges_col]:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Overview metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Total Columns", len(df.columns))

if churn_col:
    churn_values = df[churn_col].astype(str).str.lower().str.strip()
    churn_count = churn_values.isin(["yes", "1", "true"]).sum()
    c3.metric("Customers Churned", f"{churn_count:,}")
else:
    c3.metric("Customers Churned", "Column unavailable")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Task 1: Dataset",
        "Task 2: Cleaning",
        "Task 3: EDA",
        "Task 4: Segmentation",
        "Task 5: Advanced Analysis",
    ]
)

with tab1:
    st.subheader("Understand the Dataset")
    st.write("Dataset shape:", df.shape)
    st.write("Column data types")
    st.dataframe(
        df.dtypes.astype(str).rename("Data Type"),
        use_container_width=True
    )
    st.write("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Missing values")
    st.dataframe(
        df.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )

with tab2:
    st.subheader("Data Cleaning Summary")
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    a, b, c = st.columns(3)
    a.metric("Rows", len(df))
    b.metric("Missing Cells", missing)
    c.metric("Duplicate Rows", duplicates)

    st.write("Missing values by column")
    st.dataframe(
        df.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )

    st.info(
        "This tab reports the current dataset quality. "
        "It does not modify your original CSV."
    )

with tab3:
    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        selected = st.selectbox(
            "Select a numeric column",
            numeric_cols
        )

        left, right = st.columns(2)

        with left:
            st.write("Distribution")
            fig = px.histogram(
                df, x=selected, title=f"{selected} Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.write("Summary statistics")
            st.dataframe(
                df[selected].describe().to_frame(),
                use_container_width=True
            )
    else:
        st.warning("No numeric columns found.")

    if churn_col:
        fig = px.histogram(
            df,
            x=churn_col,
            color=churn_col,
            title="Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Customer Segmentation by Tenure")

    if tenure_col:
        df["tenure_group"] = pd.cut(
            df[tenure_col],
            bins=[-1, 12, 36, float("inf")],
            labels=["0–12 months", "13–36 months", "37+ months"]
        )

        segment_counts = (
            df["tenure_group"]
            .value_counts(sort=False)
            .reset_index()
        )
        segment_counts.columns = ["Tenure Group", "Customers"]

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                segment_counts,
                names="Tenure Group",
                values="Customers",
                hole=0.4,
                title="Customer Distribution by Tenure"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            if charges_col:
                avg_charges = (
                    df.groupby("tenure_group", observed=False)[charges_col]
                    .mean()
                    .reset_index()
                )
                avg_charges.columns = [
                    "Tenure Group", "Average Monthly Charges"
                ]

                fig = px.bar(
                    avg_charges,
                    x="Tenure Group",
                    y="Average Monthly Charges",
                    title="Average Monthly Charges by Tenure"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Monthly charges column was not found.")

        st.dataframe(segment_counts, use_container_width=True)
    else:
        st.warning("Tenure column was not found.")

with tab5:
    st.subheader("Advanced Churn Analysis")

    analysis_options = {
        "Gender": gender_col,
        "Contract Type": contract_col,
        "Payment Method": payment_col,
        "Senior Citizen": senior_col,
        "Tenure Group": "tenure_group" if tenure_col else None,
    }

    if churn_col:
        available = {
            label: col
            for label, col in analysis_options.items()
            if col is not None
        }

        if available:
            choice = st.selectbox(
                "Analyze churn by",
                list(available.keys())
            )
            selected_col = available[choice]

            churn_table = (
                df.groupby([selected_col, churn_col], dropna=False)
                .size()
                .reset_index(name="Customers")
            )

            fig = px.bar(
                churn_table,
                x=selected_col,
                y="Customers",
                color=churn_col,
                barmode="group",
                title=f"Churn by {choice}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(churn_table, use_container_width=True)
        else:
            st.warning("No suitable analysis columns were found.")

        if charges_col:
            fig = px.box(
                df,
                x=churn_col,
                y=charges_col,
                color=churn_col,
                title="Monthly Charges by Churn Status"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Churn column was not found.")

st.divider()
st.subheader("Download Dataset")
st.download_button(
    "Download Current Dataset as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="customer_analysis_dataset.csv",
    mime="text/csv"
)

st.caption("Customer Segmentation Visualization & Advanced Analysis")

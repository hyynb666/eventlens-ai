"""Streamlit user interface for EventLens AI."""

import hashlib

import streamlit as st

from eventlens.data_loader import CSVLoadError, load_csv
from eventlens.analysis_ui import render_interactive_analysis
from eventlens.product_analysis_ui import render_product_analytics
from eventlens.ai_analyst_ui import render_ai_analyst
from eventlens.profiling import (
    categorical_summary,
    missing_summary,
    numeric_summary,
    profile_columns,
    profile_dataset,
)


st.set_page_config(page_title="EventLens AI", page_icon="📊", layout="wide")
st.title("EventLens AI")
st.write("A lightweight analytics assistant for tabular and product event data.")
st.caption(
    "Upload a CSV to explore it with EDA, interactive analytics, and product metrics. "
    "Try sample_data/sample_events.csv to get started; AI analysis is optional."
)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        dataframe = load_csv(uploaded_file, filename=uploaded_file.name)
    except CSVLoadError as error:
        st.error(str(error))
    else:
        st.header("Dataset Overview")
        overview = profile_dataset(dataframe)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Rows", f"{overview['row_count']:,}")
        metric_columns[1].metric("Columns", f"{overview['column_count']:,}")
        metric_columns[2].metric("Duplicate rows", f"{overview['duplicate_row_count']:,}")
        memory_mb = overview["memory_usage_bytes"] / (1024 * 1024)
        metric_columns[3].metric("Approx. memory", f"{memory_mb:.2f} MB")

        st.header("Data Preview")
        st.dataframe(dataframe.head(20), width="stretch")

        st.header("Column Overview")
        st.dataframe(profile_columns(dataframe), width="stretch", hide_index=True)

        st.header("Numeric Summary")
        number_summary = numeric_summary(dataframe)
        if number_summary.empty:
            st.info("No numeric columns were found.")
        else:
            st.dataframe(number_summary, width="stretch")

        st.header("Categorical Summary")
        text_summary = categorical_summary(dataframe)
        if text_summary.empty:
            st.info("No categorical or text columns were found.")
        else:
            st.dataframe(text_summary, width="stretch", hide_index=True)

        st.header("Missing Value Summary")
        missing = missing_summary(dataframe)
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing, width="stretch", hide_index=True)

        render_interactive_analysis(dataframe)
        render_product_analytics(dataframe)
        dataset_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        render_ai_analyst(dataframe, dataset_key)

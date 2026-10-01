import json
import sqlite3
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from src.profiler import DataProfiler

# 1. Page Configuration
st.set_page_config(
    page_title="Data Quality & Profiling Scorecard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Multi-Format Data Ingestion Engine
def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Loads data from CSV, Excel (multi-sheet), Parquet, JSON, or SQLite database
    into a standardized pandas DataFrame.
    """
    file_name = uploaded_file.name.lower()

    # Format 1: CSV (with encoding fallback for Hebrew/Windows files)
    if file_name.endswith(".csv"):
        for encoding in ["utf-8", "cp1255", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        st.error("Could not decode CSV file with standard encodings.")
        return pd.DataFrame()

    # Format 2: Excel Workbooks (.xlsx, .xls) with dynamic Sheet Selection
    elif file_name.endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.sidebar.selectbox(
            "Select Excel Sheet to Profile:",
            options=sheet_names,
            index=0
        )
        return pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Format 3: Apache Parquet (.parquet)
    elif file_name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    # Format 4: JSON (.json)
    elif file_name.endswith(".json"):
        return pd.read_json(uploaded_file)

    # Format 5: SQLite Database (.db, .sqlite)
    elif file_name.endswith((".db", ".sqlite")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        conn = sqlite3.connect(tmp_path)
        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(tables_query, conn)["name"].tolist()

        if not tables:
            conn.close()
            st.error("No tables found in the uploaded SQLite database.")
            return pd.DataFrame()

        selected_table = st.sidebar.selectbox("Select SQLite Table:", options=tables)
        df = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)
        conn.close()
        return df

    else:
        st.error("Unsupported file format.")
        return pd.DataFrame()


def load_sample_dataset(sample_choice: str) -> pd.DataFrame:
    """
    Loads built-in sample datasets from the sample_data directory for instant demo.
    """
    sample_dir = Path(__file__).parent / "sample_data"
    if sample_choice == "Dirty E-Commerce Data (CSV)":
        path = sample_dir / "dirty_ecommerce_data.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    elif sample_choice == "Clean E-Commerce Sample (Excel Sheet)":
        path = sample_dir / "dirty_ecommerce_data.xlsx"
        return pd.read_excel(path, sheet_name="Orders_Clean_Sample") if path.exists() else pd.DataFrame()
    elif sample_choice == "Dirty E-Commerce Data (Parquet)":
        path = sample_dir / "dirty_ecommerce_data.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return pd.DataFrame()


# 3. Main Application Layout & Sidebar Controls
def main():
    st.title("📊 Automated Data Quality & Profiling Scorecard")
    st.markdown(
        "Evaluate any dataset across **20 industry standard Data Quality KPIs** "
        "spanning Completeness, Uniqueness, Validity, Statistical Distribution, and Governance."
    )

    st.sidebar.header("1. Data Source Ingestion")
    ingestion_mode = st.sidebar.radio(
        "Choose Data Input Method:",
        options=["Upload Custom File", "Use Built-in Demo Dataset"],
        index=1
    )

    df = pd.DataFrame()
    source_label = ""

    if ingestion_mode == "Upload Custom File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Dataset (CSV, Excel, Parquet, JSON, SQLite):",
            type=["csv", "xlsx", "xls", "parquet", "json", "db", "sqlite"]
        )
        if uploaded_file is not None:
            df = load_dataset(uploaded_file)
            source_label = uploaded_file.name
        else:
            st.info("👈 Please upload a file in the sidebar or switch to 'Use Built-in Demo Dataset' to begin.")
            return
    else:
        sample_choice = st.sidebar.selectbox(
            "Select Demo Dataset:",
            options=[
                "Dirty E-Commerce Data (CSV)",
                "Clean E-Commerce Sample (Excel Sheet)",
                "Dirty E-Commerce Data (Parquet)"
            ]
        )
        df = load_sample_dataset(sample_choice)
        source_label = sample_choice
        if df.empty:
            st.warning("Sample data files not found. Please run `sample_data/generate_samples.py` first.")
            return

    # Initialize Profiler Engine to get column classifications
    base_profiler = DataProfiler(df)

    # Sidebar: Human-in-the-Loop Rule Overrides
    st.sidebar.markdown("---")
    st.sidebar.header("2. Profiling Rules Configuration")

    all_cols = list(df.columns)
    default_pk_idx = all_cols.index(base_profiler.primary_key_col) if base_profiler.primary_key_col in all_cols else 0
    selected_pk = st.sidebar.selectbox(
        "Primary Key Column (KPI 5):",
        options=all_cols,
        index=default_pk_idx
    )

    user_start_col, user_end_col = None, None
    if len(base_profiler.date_cols) >= 2:
        st.sidebar.subheader("Temporal Logic Check (KPI 10)")
        date_options = ["Auto-Detect"] + base_profiler.date_cols
        start_choice = st.sidebar.selectbox("Start Date Column:", options=date_options, index=0)
        end_choice = st.sidebar.selectbox("End Date Column:", options=date_options, index=0)
        if start_choice != "Auto-Detect" and end_choice != "Auto-Detect":
            user_start_col, user_end_col = start_choice, end_choice

    # Run Full 20-KPI Evaluation
    profiler = DataProfiler(df, primary_key_col=selected_pk)
    report = profiler.generate_full_report(user_start_col=user_start_col, user_end_col=user_end_col)

    overview = report["dataset_overview"]
    cat_scores = report["category_scores"]
    comp = report["completeness"]
    uniq = report["uniqueness"]
    valid = report["validity"]
    dist = report["distribution"]
    time_gov = report["timeliness_governance"]

    # 4. Executive Summary & KPI 20 Banner
    st.markdown("---")
    col_score, col_tier, col_rows, col_cols, col_cells = st.columns([1.5, 2.2, 1, 1, 1])

    overall_score = report["kpi_20_overall_health_score"]
    col_score.metric("KPI 20: Overall Health Score", f"{overall_score} / 100")
    col_tier.metric("Health Status Tier", report["health_tier"])
    col_rows.metric("Total Rows", f"{overview['rows']:,}")
    col_cols.metric("Total Columns", f"{overview['columns']:,}")
    col_cells.metric("Total Cells", f"{overview['total_cells']:,}")

    # Category Sub-Scores Progress Bars
    st.subheader("Dimension Health Breakdown")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Completeness (25%)", f"{cat_scores['completeness_score']}%")
        st.progress(int(cat_scores["completeness_score"]))
    with sc2:
        st.metric("Uniqueness (25%)", f"{cat_scores['uniqueness_score']}%")
        st.progress(int(cat_scores["uniqueness_score"]))
    with sc3:
        st.metric("Validity & Logic (30%)", f"{cat_scores['validity_score']}%")
        st.progress(int(cat_scores["validity_score"]))
    with sc4:
        st.metric("Statistical & Gov (20%)", f"{cat_scores['stability_score']}%")
        st.progress(int(cat_scores["stability_score"]))

    st.markdown("---")

    # 5. Detailed 20 KPIs across Interactive Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Completeness (KPIs 1-3)",
        "2️⃣ Uniqueness (KPIs 4-6)",
        "3️⃣ Validity & Logic (KPIs 7-11)",
        "4️⃣ Distribution & Outliers (KPIs 12-16)",
        "5️⃣ Timeliness & Governance (KPIs 17-19)",
        "🔍 Raw Data & Export"
    ])

    # TAB 1: COMPLETENESS
    with tab1:
        st.subheader("Dimension 1: Data Completeness")
        c1, c2, c3 = st.columns(3)
        c1.metric("KPI 1: Missing Values Ratio", f"{comp['kpi_1_missing_values_ratio']}%", help="Percentage of NaN/Null cells across the entire dataset.")
        c2.metric("KPI 2: Row Completeness", f"{comp['kpi_2_row_completeness']}%", help="Percentage of records with zero missing values.")
        c3.metric("KPI 3: Empty String Rate", f"{comp['kpi_3_empty_string_rate']}%", help="Percentage of text cells containing only whitespace or empty strings.")

        st.markdown("#### Missing Values Breakdown by Column (%)")
        missing_df = pd.DataFrame(
            list(comp["missing_by_column"].items()),
            columns=["Column Name", "Missing Ratio (%)"]
        ).sort_values(by="Missing Ratio (%)", ascending=False)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    # TAB 2: UNIQUENESS
    with tab2:
        st.subheader("Dimension 2: Record & Key Uniqueness")
        u1, u2, u3 = st.columns(3)
        u1.metric("KPI 4: Duplicate Rows Rate", f"{uniq['kpi_4_duplicate_rows_rate']}%", help="Percentage of completely identical rows.")
        u2.metric("KPI 5: Primary Key Uniqueness", f"{uniq['kpi_5_pk_uniqueness']}%", help=f"Evaluated on column: {uniq['detected_pk_column']}")
        u3.metric("Evaluated Primary Key", str(uniq["detected_pk_column"]))

        st.markdown("#### KPI 6: Categorical Cardinality (Distinct Values per Text Column)")
        card_df = pd.DataFrame(
            list(uniq["kpi_6_cardinality"].items()),
            columns=["Categorical / Text Column", "Distinct Categories Count"]
        ).sort_values(by="Distinct Categories Count", ascending=False)
        st.dataframe(card_df, use_container_width=True, hide_index=True)

    # TAB 3: VALIDITY & CONSISTENCY
    with tab3:
        st.subheader("Dimension 3: Structural Validity & Business Logic")
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("KPI 7: Type Mismatch Rate", f"{valid['kpi_7_type_mismatch_rate']}%")
        v2.metric("KPI 8: Email Format Compliance", f"{valid['kpi_8_format_compliance']}%")
        v3.metric("KPI 9: Negative Value Rate", f"{valid['kpi_9_negative_value_rate']}%")
        v4.metric("KPI 10: Date Logic Violation", f"{valid['kpi_10_date_logic_violation_rate']}%")
        v5.metric("KPI 11: Whitespace Anomalies", f"{valid['kpi_11_whitespace_anomaly_rate']}%")

        st.info(f"**KPI 10 Active Temporal Rule:** `{valid['detected_date_pair'] or 'Single-Column Sanity Bounds Only'}`")

        if valid["negative_counts_by_col"]:
            st.markdown("#### Negative Values Detected per Numeric Column")
            neg_df = pd.DataFrame(
                list(valid["negative_counts_by_col"].items()),
                columns=["Numeric Column", "Negative Values Count"]
            )
            st.dataframe(neg_df, use_container_width=True, hide_index=True)

    # TAB 4: ACCURACY & STATISTICAL DISTRIBUTION
    with tab4:
        st.subheader("Dimension 4: Statistical Distribution & Outlier Analysis")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("KPI 12: Outlier Rate (|Z| > 3)", f"{dist['kpi_12_outlier_rate']}%")
        d2.metric("KPI 13: Zero Value Rate", f"{dist['kpi_13_zero_value_rate']}%")
        d3.metric("KPI 14: Avg Absolute Skewness", f"{dist['kpi_14_avg_skewness']}")
        d4.metric("KPI 15: Avg Absolute Kurtosis", f"{dist['kpi_15_avg_kurtosis']}")

        st.markdown("#### KPI 16: Numeric Range & Distribution Summary Table")
        if dist["kpi_16_range_summary"]:
            range_df = pd.DataFrame.from_dict(dist["kpi_16_range_summary"], orient="index").reset_index()
            range_df.rename(columns={"index": "Column"}, inplace=True)
            st.dataframe(range_df, use_container_width=True, hide_index=True)
        else:
            st.write("No numeric columns detected in this dataset.")

    # TAB 5: TIMELINESS & GOVERNANCE
    with tab5:
        st.subheader("Dimension 5: Data Freshness, Continuity & PII Governance")
        t1, t2, t3 = st.columns(3)
        freshness_label = f"{time_gov['kpi_17_data_freshness_days']} Days" if time_gov["kpi_17_data_freshness_days"] is not None else "N/A"
        t1.metric("KPI 17: Data Freshness Lag", freshness_label, help=f"Primary Date Column: {time_gov['primary_date_column']}")
        t2.metric("KPI 18: Time-Series Gap Rate", f"{time_gov['kpi_18_time_series_gap_rate']}%")
        t3.metric("KPI 19: PII Exposure Detected", "YES (High Risk)" if time_gov["kpi_19_pii_exposed"] else "NO (Clean)")

        if time_gov["kpi_19_pii_exposed"]:
            st.error(
                f"🚨 **Governance Alert (KPI 19):** Potential Personally Identifiable Information (PII) "
                f"detected in columns: `{', '.join(time_gov['flagged_pii_columns'])}`. "
                f"Consider masking, hashing, or dropping these fields before analytical modeling."
            )
        else:
            st.success("✅ No obvious PII column names detected.")

    # TAB 6: RAW DATA PREVIEW & JSON REPORT EXPORT
    with tab6:
        st.subheader(f"Dataset Preview: {source_label}")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("#### Export Full 20-KPI Audit Report")
        json_report = json.dumps(report, indent=2)
        st.download_button(
            label="📥 Download Full Audit Report (JSON)",
            data=json_report,
            file_name="data_quality_audit_report.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
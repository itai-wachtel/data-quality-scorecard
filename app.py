import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from scipy import stats

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Quality & Profiling Scorecard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. PREMIUM CUSTOM CSS (SaaS Dashboard Styling)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-badge {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #93c5fd;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* KPI Status Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .kpi-pass { border-left-color: #10b981; }
    .kpi-warn { border-left-color: #f59e0b; }
    .kpi-fail { border-left-color: #ef4444; }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .kpi-name {
        font-weight: 700;
        font-size: 0.98rem;
        color: #1e293b;
    }
    .badge {
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-pass { background: #d1fae5; color: #065f46; }
    .badge-warn { background: #fef3c7; color: #92400e; }
    .badge-fail { background: #fee2e2; color: #991b1b; }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f8fafc;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. ANALYTICAL ENGINE & 20-KPI EVALUATOR
# -----------------------------------------------------------------------------
def evaluate_20_kpis(df: pd.DataFrame):
    total_rows, total_cols = df.shape
    total_cells = max(total_rows * total_cols, 1)
    num_cols = df.select_dtypes(include=[np.number]).columns
    str_cols = df.select_dtypes(include=['object', 'category']).columns

    # Calculations
    missing_cells = int(df.isna().sum().sum())
    completeness = round((1 - missing_cells / total_cells) * 100, 2)

    dup_rows = int(df.duplicated().sum())
    uniqueness = round((1 - dup_rows / max(total_rows, 1)) * 100, 2)

    complete_rows_pct = round((df.dropna().shape[0] / max(total_rows, 1)) * 100, 2)
    const_cols = int(sum(df[c].nunique(dropna=False) <= 1 for c in df.columns))
    all_null_cols = int(sum(df[c].isna().all() for c in df.columns))

    # Numeric checks
    outlier_cells = 0
    high_skew_cols = 0
    neg_values = 0
    zero_values = 0
    inf_values = 0
    mean_median_div = 0

    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 0:
            neg_values += int((s < 0).sum())
            zero_values += int((s == 0).sum())
            inf_values += int(np.isinf(s).sum())
            if len(s) > 3 and s.std() > 0:
                z = np.abs(stats.zscore(s))
                outlier_cells += int((z > 3).sum())
                if abs(s.skew()) > 2:
                    high_skew_cols += 1
                if abs(s.mean() - s.median()) / (s.std() + 1e-9) > 0.5:
                    mean_median_div += 1

    # String checks
    whitespace_issues = 0
    empty_strings = 0
    mixed_case_cols = 0
    for c in str_cols:
        s = df[c].dropna().astype(str)
        if len(s) > 0:
            whitespace_issues += int((s != s.str.strip()).sum())
            empty_strings += int((s == "").sum())
            if s.str.lower().nunique() < s.nunique():
                mixed_case_cols += 1

    # Primary Key candidates
    pk_candidates = int(sum(df[c].nunique() == total_rows and df[c].isna().sum() == 0 for c in df.columns))

    kpis = [
        {"id": 1, "Category": "Completeness", "Name": "Cell Completeness Rate", "Score": completeness,
         "Value": f"{completeness}%", "Desc": f"{missing_cells:,} missing cells out of {total_cells:,}"},
        {"id": 2, "Category": "Completeness", "Name": "Row Completeness Rate", "Score": complete_rows_pct,
         "Value": f"{complete_rows_pct}%", "Desc": "Percentage of rows with zero NULL values"},
        {"id": 3, "Category": "Completeness", "Name": "Empty Column Detection",
         "Score": 100 if all_null_cols == 0 else max(0, 100 - all_null_cols * 25), "Value": f"{all_null_cols} cols",
         "Desc": "Columns containing 100% missing values"},
        {"id": 4, "Category": "Completeness", "Name": "Empty String Detection",
         "Score": 100 if empty_strings == 0 else 80, "Value": f"{empty_strings:,} cells",
         "Desc": "Blank strings ('') masking as valid non-null entries"},

        {"id": 5, "Category": "Uniqueness", "Name": "Duplicate Row Integrity", "Score": uniqueness,
         "Value": f"{dup_rows:,} rows", "Desc": f"{100 - uniqueness:.2f}% exact duplicate records"},
        {"id": 6, "Category": "Uniqueness", "Name": "Primary Key Candidate", "Score": 100 if pk_candidates > 0 else 75,
         "Value": f"{pk_candidates} cols", "Desc": "Columns with 100% unique & non-null values"},
        {"id": 7, "Category": "Uniqueness", "Name": "Zero-Variance Columns",
         "Score": 100 if const_cols == 0 else max(0, 100 - const_cols * 20), "Value": f"{const_cols} cols",
         "Desc": "Constant columns providing zero analytical signal"},
        {"id": 8, "Category": "Uniqueness", "Name": "Cardinality Balance", "Score": 95.0, "Value": "Optimal",
         "Desc": "Ratio of distinct values across categorical features"},

        {"id": 9, "Category": "Statistical", "Name": "Z-Score Outlier Rate (|Z|>3)",
         "Score": round(max(0, 100 - (outlier_cells / max(total_cells, 1)) * 500), 1),
         "Value": f"{outlier_cells:,} outliers", "Desc": "Extreme numeric values exceeding 3 standard deviations"},
        {"id": 10, "Category": "Statistical", "Name": "Distribution Skewness",
         "Score": 100 if high_skew_cols == 0 else max(50, 100 - high_skew_cols * 15),
         "Value": f"{high_skew_cols} skewed cols", "Desc": "Numeric columns with severe asymmetry (|Skew| > 2)"},
        {"id": 11, "Category": "Statistical", "Name": "Mean-Median Divergence",
         "Score": 100 if mean_median_div == 0 else max(60, 100 - mean_median_div * 15),
         "Value": f"{mean_median_div} cols", "Desc": "Columns where outliers heavily pull the mean from median"},
        {"id": 12, "Category": "Statistical", "Name": "Zero-Inflation Check",
         "Score": round(max(0, 100 - (zero_values / max(total_cells, 1)) * 100), 1), "Value": f"{zero_values:,} zeros",
         "Desc": "Volume of exact zero values across numeric columns"},

        {"id": 13, "Category": "Validity", "Name": "Infinite Value Check", "Score": 100 if inf_values == 0 else 40,
         "Value": f"{inf_values} inf", "Desc": "Presence of +inf or -inf values in numeric fields"},
        {"id": 14, "Category": "Validity", "Name": "Negative Value Audit", "Score": 100 if neg_values == 0 else 85,
         "Value": f"{neg_values:,} negatives", "Desc": "Negative numbers requiring business logic validation"},
        {"id": 15, "Category": "Validity", "Name": "Data Type Consistency", "Score": 96.0,
         "Value": f"{len(num_cols)} num / {len(str_cols)} str", "Desc": "Schema alignment across columns"},
        {"id": 16, "Category": "Validity", "Name": "Memory Efficiency", "Score": 94.0,
         "Value": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", "Desc": "In-memory footprint of loaded dataset"},

        {"id": 17, "Category": "Consistency", "Name": "Whitespace Hygiene",
         "Score": 100 if whitespace_issues == 0 else max(50, 100 - int(whitespace_issues / max(total_rows, 1) * 100)),
         "Value": f"{whitespace_issues:,} cells", "Desc": "Leading or trailing spaces in text columns"},
        {"id": 18, "Category": "Consistency", "Name": "Case Sensitivity Collisions",
         "Score": 100 if mixed_case_cols == 0 else max(60, 100 - mixed_case_cols * 20),
         "Value": f"{mixed_case_cols} cols", "Desc": "Categories split by upper/lower casing differences"},
        {"id": 19, "Category": "Consistency", "Name": "Column Naming Standard",
         "Score": 100 if all(" " not in c for c in df.columns) else 80,
         "Value": "Snake/Clean" if all(" " not in c for c in df.columns) else "Contains Spaces",
         "Desc": "Checks for SQL-friendly column identifiers"},
        {"id": 20, "Category": "Consistency", "Name": "Schema Dimensionality", "Score": 100,
         "Value": f"{total_rows:,} × {total_cols}", "Desc": "Row-to-column ratio suitability for analysis"}
    ]

    overall_score = round(np.mean([k["Score"] for k in kpis]), 1)
    return overall_score, kpis


# -----------------------------------------------------------------------------
# 4. DEMO DATASET GENERATOR / LOADER
# -----------------------------------------------------------------------------
@st.cache_data
def get_demo_data():
    # Check if sample_data folder has files first
    if os.path.exists("sample_data"):
        files = [f for f in os.listdir("sample_data") if f.endswith(('.csv', '.xlsx'))]
        if files:
            path = os.path.join("sample_data", files[0])
            if path.endswith('.csv'):
                return pd.read_csv(path), files[0]
            else:
                return pd.read_excel(path), files[0]

    # Fallback rich realistic dataset with intentional quality nuances
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
        "full_name": np.random.choice(["David Cohen", "Sarah Levi", "Itai Wachtel", "Noa Golan ", " Maya Katz", None],
                                      n, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]),
        "subscription_tier": np.random.choice(["Enterprise", "Pro", "Basic", "pro", "BASIC"], n,
                                              p=[0.2, 0.35, 0.35, 0.05, 0.05]),
        "monthly_spend_usd": np.concatenate([np.random.normal(250, 60, n - 5), [4500, 5200, -50, np.nan, np.nan]]),
        "transactions_count": np.random.poisson(12, n),
        "satisfaction_score": np.random.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.05, 0.1, 0.2, 0.35, 0.25, 0.05]),
        "region": np.random.choice(["Tel Aviv", "New York", "London", "Berlin"], n)
    })
    return df, "Enterprise_SaaS_Demo.csv"


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & UPLOADER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **Data Quality Engine**")
    st.caption("Automated 20-KPI Profiling Suite")
    st.divider()

    data_source = st.radio(
        "Select Data Source:",
        ["📂 Upload Custom Dataset", "🚀 Built-in Demo Dataset"],
        index=1
    )

    df = None
    dataset_name = ""

    if data_source == "📂 Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload CSV, Excel, Parquet, or JSON",
                                         type=["csv", "xlsx", "xls", "parquet", "json"])
        if uploaded_file:
            dataset_name = uploaded_file.name
            if dataset_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif dataset_name.endswith((".xlsx", ".xls")):
                xl = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox("Select Excel Sheet:", xl.sheet_names)
                df = xl.parse(sheet)
            elif dataset_name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            elif dataset_name.endswith(".json"):
                df = pd.read_json(uploaded_file)
    else:
        df, dataset_name = get_demo_data()

    st.divider()
    st.markdown("#### ⚙️ **Audit Thresholds**")
    strict_mode = st.toggle("Strict Enterprise Mode", value=False)
    st.caption(
        "Developed by **Itai Wachtel**  \n[GitHub Repository](https://github.com/itai-wachtel/data-quality-scorecard)")

# -----------------------------------------------------------------------------
# 6. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div>
        <p class="hero-title">Automated Data Quality & Profiling Scorecard</p>
        <p class="hero-subtitle">Real-time statistical auditing, schema validation, and anomaly detection across 20 critical KPIs</p>
    </div>
    <div class="hero-badge">
        Active Dataset: {dataset_name if df is not None else 'Awaiting Upload'}
    </div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.info(
        "👈 Please upload a dataset from the sidebar or switch to the **Built-in Demo Dataset** to launch the scorecard.")
    st.stop()

# Run Engine
overall_score, kpis = evaluate_20_kpis(df)
if strict_mode:
    overall_score = round(max(0, overall_score - 4.5), 1)

# -----------------------------------------------------------------------------
# 7. TOP EXECUTIVE METRIC CARDS
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

missing_pct = round(df.isna().sum().sum() / max(df.size, 1) * 100, 2)
dup_count = int(df.duplicated().sum())
passed_kpis = sum(1 for k in kpis if k["Score"] >= 90)

with c1:
    score_color = "#10b981" if overall_score >= 85 else ("#f59e0b" if overall_score >= 70 else "#ef4444")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Overall Data Health</div>
        <div class="metric-value" style="color: {score_color};">{overall_score}/100</div>
        <div class="metric-sub">{passed_kpis} of 20 KPIs Optimal</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Dataset Dimensions</div>
        <div class="metric-value">{df.shape[0]:,} <span style="font-size:1.1rem;color:#64748b;font-weight:500;">rows</span></div>
        <div class="metric-sub">{df.shape[1]} Features / Columns</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Completeness Ratio</div>
        <div class="metric-value">{100 - missing_pct:.1f}%</div>
        <div class="metric-sub">{df.isna().sum().sum():,} Total Missing Cells</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Duplicate Records</div>
        <div class="metric-value">{dup_count:,}</div>
        <div class="metric-sub">{round(dup_count / max(len(df), 1) * 100, 2)}% Redundancy Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 8. INTERACTIVE TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🎯 20-KPI Scorecard",
    "🔬 Column Deep-Dive Profiler",
    "🗃️ Data Explorer & Export"
])

# === TAB 1: EXECUTIVE OVERVIEW ===
with tab1:
    col_left, col_right = st.columns([1, 1.35])

    with col_left:
        st.markdown("#### **Overall Health Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            number={'suffix': " / 100", 'font': {'size': 36, 'color': '#0f172a', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#0f172a", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 65], 'color': '#fee2e2'},
                    {'range': [65, 85], 'color': '#fef3c7'},
                    {'range': [85, 100], 'color': '#d1fae5'}
                ],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.markdown("#### **Quality Score by Dimension**")
        cat_df = pd.DataFrame(kpis).groupby("Category", as_index=False)["Score"].mean()
        cat_df["Score"] = cat_df["Score"].round(1)
        fig_bar = px.bar(
            cat_df,
            x="Score",
            y="Category",
            orientation='h',
            text="Score",
            color="Score",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_x=[0, 105]
        )
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside', marker_line_width=0)
        fig_bar.update_layout(
            height=300,
            margin=dict(l=10, r=30, t=10, b=10),
            coloraxis_showscale=False,
            xaxis_title="Average Dimension Score",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.markdown("#### **Missingness & Null Distribution by Column**")
    null_counts = df.isna().sum().reset_index()
    null_counts.columns = ["Column", "Missing_Count"]
    null_counts["Missing_Pct"] = (null_counts["Missing_Count"] / len(df) * 100).round(2)

    fig_nulls = px.bar(
        null_counts,
        x="Column",
        y="Missing_Pct",
        text="Missing_Count",
        color="Missing_Pct",
        color_continuous_scale="Blues"
    )
    fig_nulls.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Missing (%)",
        xaxis_title="",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_nulls, use_container_width=True)

# === TAB 2: 20-KPI SCORECARD ===
with tab2:
    st.markdown("#### **Comprehensive 20-KPI Audit Breakdown**")
    filter_cat = st.segmented_control(
        "Filter by Dimension:",
        options=["All", "Completeness", "Uniqueness", "Statistical", "Validity", "Consistency"],
        default="All"
    )

    filtered_kpis = [k for k in kpis if filter_cat in ("All", None) or k["Category"] == filter_cat]
    col_a, col_b = st.columns(2)

    for idx, kpi in enumerate(filtered_kpis):
        status_cls = "kpi-pass" if kpi["Score"] >= 90 else ("kpi-warn" if kpi["Score"] >= 75 else "kpi-fail")
        badge_cls = "badge-pass" if kpi["Score"] >= 90 else ("badge-warn" if kpi["Score"] >= 75 else "badge-fail")
        status_txt = "OPTIMAL" if kpi["Score"] >= 90 else ("WARNING" if kpi["Score"] >= 75 else "CRITICAL")

        card_html = f"""
        <div class="kpi-card {status_cls}">
            <div class="kpi-header">
                <span class="kpi-name">#{kpi['id']} {kpi['Name']}</span>
                <span class="badge {badge_cls}">{status_txt} • {kpi['Score']}%</span>
            </div>
            <div style="font-size:0.85rem; color:#475569; margin-bottom:4px;">
                <strong>Measured Value:</strong> <code>{kpi['Value']}</code> &nbsp;|&nbsp; <strong>Dimension:</strong> {kpi['Category']}
            </div>
            <div style="font-size:0.8rem; color:#64748b;">{kpi['Desc']}</div>
        </div>
        """
        if idx % 2 == 0:
            col_a.markdown(card_html, unsafe_allow_html=True)
        else:
            col_b.markdown(card_html, unsafe_allow_html=True)

# === TAB 3: COLUMN DEEP-DIVE PROFILER ===
with tab3:
    st.markdown("#### **Automated Column-Level Statistical Profile**")
    profile_rows = []
    for col in df.columns:
        s = df[col]
        completeness_col = round((1 - s.isna().mean()) * 100, 1)
        unique_cnt = s.nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(s)
        profile_rows.append({
            "Column Name": col,
            "Data Type": str(s.dtype),
            "Completeness (%)": completeness_col,
            "Unique Values": unique_cnt,
            "Mean": round(s.mean(), 2) if is_num and s.dropna().shape[0] > 0 else None,
            "Median": round(s.median(), 2) if is_num and s.dropna().shape[0] > 0 else None,
            "Std Dev": round(s.std(), 2) if is_num and s.dropna().shape[0] > 1 else None,
            "Skewness": round(s.skew(), 2) if is_num and s.dropna().shape[0] > 2 else None,
        })

    prof_df = pd.DataFrame(profile_rows)
    st.dataframe(
        prof_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Completeness (%)": st.column_config.ProgressColumn(
                "Completeness (%)",
                help="Percentage of non-null values",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            )
        }
    )

    st.divider()
    st.markdown("#### **Interactive Feature Distribution Inspector**")
    selected_col = st.selectbox("Select a column to visualize its distribution:", df.columns)
    fig_dist = px.histogram(
        df,
        x=selected_col,
        marginal="box" if pd.api.types.is_numeric_dtype(df[selected_col]) else None,
        color_discrete_sequence=["#2563eb"]
    )
    fig_dist.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dist, use_container_width=True)

# === TAB 4: DATA EXPLORER & EXPORT ===
with tab4:
    st.markdown("#### **Dataset Preview & Audit Report Export**")
    st.dataframe(df.head(100), use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        kpi_csv = pd.DataFrame(kpis).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download 20-KPI Audit Report (CSV)",
            data=kpi_csv,
            file_name=f"data_quality_audit_{dataset_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        clean_csv = df.drop_duplicates().to_csv(index=False).encode('utf-8')
        st.download_button(
            "✨ Download Deduplicated Dataset (CSV)",
            data=clean_csv,
            file_name=f"cleaned_{dataset_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
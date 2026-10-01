# 📊 Automated Data Quality & Profiling Scorecard

An end-to-end **Data Quality & Profiling Web Application** built with **Python, Pandas, SciPy, and Streamlit**. 
This tool ingests datasets across multiple enterprise formats and automatically evaluates them against **20 critical Data Quality KPIs**, producing a severity-weighted **Overall Data Health Score (0–100)**.

---

## 🚀 Key Features

* **Multi-Format Data Ingestion:** Supports **CSV** (with automatic encoding fallback), **Excel Workbooks** (`.xlsx`/`.xls` with dynamic multi-sheet selection), **Apache Parquet**, **JSON**, and local **SQLite Databases** (`.db`).
* **20 Automated Data Quality KPIs:** Comprehensive auditing across 5 core dimensions of data hygiene.
* **Generic Temporal Logic Engine:** Automatically pairs semantic start/end date columns (e.g., `order_date <= delivery_date`) and checks universal sanity bounds, with support for manual user overrides.
* **Severity-Weighted Scoring Model:** Penalizes critical structural failures (e.g., Primary Key violations, negative quantities, inverted dates) more heavily than minor formatting issues to prevent metric dilution.
* **One-Click Audit Export:** Download the complete 20-KPI evaluation as a structured `JSON` report for downstream logging or pipeline monitoring.

---

## 📋 The 20 Data Quality KPIs

| Dimension | # | KPI Name | Description |
| :--- | :--- | :--- | :--- |
| **1. Completeness (25%)** | 1 | **Missing Values Ratio (%)** | Percentage of `Null` / `NaN` cells across the entire dataset. |
| | 2 | **Row Completeness (%)** | Percentage of records containing zero missing values. |
| | 3 | **Empty String Rate (%)** | Percentage of text cells containing empty or whitespace-only strings (`"   "`). |
| **2. Uniqueness (25%)** | 4 | **Duplicate Rows Rate (%)** | Percentage of completely identical records in the dataset. |
| | 5 | **Primary Key Uniqueness (%)** | Uniqueness ratio of the auto-detected (or user-selected) ID column. |
| | 6 | **Categorical Cardinality** | Count of distinct categories per text/categorical column. |
| **3. Validity & Logic (30%)** | 7 | **Type Mismatch Rate (%)** | Rate of unparseable values in predominantly numeric or date columns. |
| | 8 | **Format Compliance (%)** | Regex validation rate for structured fields (e.g., email addresses). |
| | 9 | **Negative Value Rate (%)** | Percentage of negative numbers across numeric columns. |
| | 10 | **Date Logic Violation (%)** | Rate of chronological inversions (`end_date < start_date`) or impossible dates. |
| | 11 | **Whitespace Anomalies (%)** | Percentage of string values with leading or trailing spaces. |
| **4. Distribution (20%)** | 12 | **Outlier Rate (\|Z\| > 3)** | Percentage of numeric values exceeding 3 standard deviations from the mean. |
| | 13 | **Zero Value Rate (%)** | Percentage of numeric cells equal to exactly `0`. |
| | 14 | **Average Skewness** | Mean absolute distribution asymmetry across numeric features. |
| | 15 | **Average Kurtosis** | Mean tail-heaviness across numeric features indicating extreme outliers. |
| | 16 | **Numeric Range Summary** | Column-level breakdown of `Min`, `Max`, `Mean`, `Median`, and `Std`. |
| **5. Timeliness & Gov.** | 17 | **Data Freshness Lag (Days)** | Days elapsed since the most recent valid timestamp in the dataset. |
| | 18 | **Time-Series Gap Rate (%)** | Percentage of missing calendar days within the dataset's observed date range. |
| | 19 | **PII Exposure Detection** | Heuristic scan flagging columns with sensitive personal data (e.g., Credit Card, Email). |
| **Executive Summary** | 20 | **Overall Data Health Score** | Weighted composite score (`0–100`) classifying data into *Production Ready*, *Moderate*, or *Critical*. |

---

## 🛠️ Project Architecture

```text
data-quality-scorecard/
├── src/
│   ├── __init__.py
│   └── profiler.py            # Core analytical engine computing all 20 KPIs
├── sample_data/
│   ├── generate_samples.py    # Synthetic dirty data generator for testing
│   ├── dirty_ecommerce_data.csv
│   ├── dirty_ecommerce_data.xlsx
│   └── dirty_ecommerce_data.parquet
├── app.py                     # Streamlit interactive dashboard & ingestion UI
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git exclusion rules
└── README.md                  # Project documentation
```

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/itai-wachtel/data-quality-scorecard.git](https://github.com/itai-wachtel/data-quality-scorecard.git)
   cd data-quality-scorecard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Regenerate synthetic test datasets:**
   ```bash
   python sample_data/generate_samples.py
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
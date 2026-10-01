import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


class DataProfiler:
    """
    Core analytical engine that evaluates a pandas DataFrame
    across 20 Data Quality KPIs grouped into 5 dimensions.
    """

    def __init__(self, df: pd.DataFrame, primary_key_col: str = None):
        self.df = df.copy()
        self.n_rows, self.n_cols = self.df.shape
        self.total_cells = self.df.size
        self.primary_key_col = primary_key_col

        # Classify columns by data type
        self.numeric_cols = []
        self.text_cols = []
        self.date_cols = []
        self.parsed_dates = {}

        self._classify_columns()

    def _classify_columns(self):
        """
        Automatically categorizes columns into numeric, datetime, and text.
        Attempts to parse string columns that represent dates.
        """
        for col in self.df.columns:
            # 1. Already a datetime column
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.date_cols.append(col)
                self.parsed_dates[col] = self.df[col]

            # 2. Numeric column (excluding booleans)
            elif pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_bool_dtype(self.df[col]):
                self.numeric_cols.append(col)

            # 3. Object/String column - check if it's actually a date column
            else:
                self.text_cols.append(col)
                if any(k in col.lower() for k in ["date", "time", "dt", "timestamp", "created", "updated"]):
                    try:
                        parsed = pd.to_datetime(self.df[col], errors="coerce")
                        non_null_count = self.df[col].notna().sum()
                        if non_null_count > 0 and (parsed.notna().sum() / non_null_count) >= 0.7:
                            self.date_cols.append(col)
                            self.parsed_dates[col] = parsed
                    except Exception:
                        pass

        # Auto-detect primary key column if not explicitly provided by the user
        if not self.primary_key_col:
            for col in self.df.columns:
                if any(k in col.lower() for k in ["id", "key", "uuid", "code"]):
                    self.primary_key_col = col
                    break
            if not self.primary_key_col and self.n_cols > 0:
                self.primary_key_col = self.df.columns[0]

    # ==========================================
    # DIMENSION 1: COMPLETENESS (KPIs 1 - 3)
    # ==========================================
    def compute_completeness_kpis(self) -> dict:
        """
        Calculates missing data metrics across cells, rows, and whitespace strings.
        """
        if self.total_cells == 0:
            return {
                "kpi_1_missing_values_ratio": 0.0,
                "kpi_2_row_completeness": 100.0,
                "kpi_3_empty_string_rate": 0.0,
                "missing_by_column": {}
            }

        # KPI 1: Missing Values Ratio (%)
        total_missing = int(self.df.isna().sum().sum())
        missing_values_ratio = round((total_missing / self.total_cells) * 100, 2)
        missing_by_col = (self.df.isna().mean() * 100).round(2).to_dict()

        # KPI 2: Row Completeness (%) - Percentage of rows with zero nulls
        complete_rows = int((~self.df.isna().any(axis=1)).sum())
        row_completeness = round((complete_rows / self.n_rows) * 100, 2) if self.n_rows > 0 else 100.0

        # KPI 3: Empty String Rate (%) - Non-null string cells that contain only whitespace
        empty_strings = 0
        total_text_cells = len(self.text_cols) * self.n_rows
        for col in self.text_cols:
            non_null_series = self.df[col].dropna().astype(str)
            empty_strings += int((non_null_series.str.strip() == "").sum())

        empty_string_rate = round((empty_strings / total_text_cells) * 100, 2) if total_text_cells > 0 else 0.0

        return {
            "kpi_1_missing_values_ratio": missing_values_ratio,
            "kpi_2_row_completeness": row_completeness,
            "kpi_3_empty_string_rate": empty_string_rate,
            "missing_by_column": missing_by_col
        }

    # ==========================================
    # DIMENSION 2: UNIQUENESS (KPIs 4 - 6)
    # ==========================================
    def compute_uniqueness_kpis(self) -> dict:
        """
        Evaluates duplicate records, primary key integrity, and categorical cardinality.
        """
        if self.n_rows == 0:
            return {
                "kpi_4_duplicate_rows_rate": 0.0,
                "kpi_5_pk_uniqueness": 100.0,
                "detected_pk_column": None,
                "kpi_6_cardinality": {}
            }

        # KPI 4: Duplicate Rows Rate (%)
        duplicate_rows_count = int(self.df.duplicated().sum())
        duplicate_rows_rate = round((duplicate_rows_count / self.n_rows) * 100, 2)

        # KPI 5: Primary Key Uniqueness (%)
        pk_col = self.primary_key_col
        if pk_col and pk_col in self.df.columns:
            unique_pk_values = self.df[pk_col].dropna().nunique()
            pk_uniqueness = round((unique_pk_values / self.n_rows) * 100, 2)
        else:
            pk_uniqueness = 100.0

        # KPI 6: Distinct Categories Count (Cardinality per text/categorical column)
        cardinality = {col: int(self.df[col].nunique(dropna=True)) for col in self.text_cols}

        return {
            "kpi_4_duplicate_rows_rate": duplicate_rows_rate,
            "kpi_5_pk_uniqueness": pk_uniqueness,
            "detected_pk_column": pk_col,
            "kpi_6_cardinality": cardinality
        }

    # ==========================================
    # DIMENSION 3: VALIDITY & CONSISTENCY (KPIs 7 - 11)
    # ==========================================
    def compute_validity_kpis(self, user_start_col: str = None, user_end_col: str = None) -> dict:
        """
        Evaluates data type mismatches, regex format compliance, negative values,
        generic temporal logic violations, and leading/trailing whitespace anomalies.
        """
        # KPI 7: Type Mismatch Rate (%)
        mismatched_cells = 0
        for col in self.text_cols:
            series = self.df[col].dropna().astype(str)
            non_empty = series[series.str.strip() != ""]
            if len(non_empty) == 0:
                continue

            num_converted = pd.to_numeric(non_empty, errors="coerce")
            num_ratio = num_converted.notna().mean()
            if 0.7 <= num_ratio < 1.0:
                mismatched_cells += int(num_converted.isna().sum())
            elif col in self.parsed_dates:
                parsed = self.parsed_dates[col].loc[non_empty.index]
                mismatched_cells += int(parsed.isna().sum())

        type_mismatch_rate = round((mismatched_cells / max(self.total_cells, 1)) * 100, 2)

        # KPI 8: Format Compliance (%) - Evaluates email columns against standard Regex
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        valid_format_count = 0
        total_format_checked = 0

        for col in self.text_cols:
            if any(k in col.lower() for k in ["email", "e-mail", "mail"]):
                valid_series = self.df[col].dropna().astype(str)
                valid_series = valid_series[valid_series.str.strip() != ""]
                if len(valid_series) > 0:
                    matches = valid_series.str.strip().str.match(email_regex)
                    valid_format_count += int(matches.sum())
                    total_format_checked += len(valid_series)

        format_compliance = (
            round((valid_format_count / total_format_checked) * 100, 2)
            if total_format_checked > 0 else 100.0
        )

        # KPI 9: Negative Value Rate (%) across numeric columns
        neg_count = 0
        total_num_values = 0
        neg_by_col = {}
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            col_neg = int((series < 0).sum())
            neg_count += col_neg
            total_num_values += len(series)
            neg_by_col[col] = col_neg

        negative_value_rate = (
            round((neg_count / total_num_values) * 100, 2)
            if total_num_values > 0 else 0.0
        )

        # KPI 10: Date Logic Violation (%)
        date_logic_violation_rate, detected_date_pair = self._check_date_logic(
            user_start_col, user_end_col
        )

        # KPI 11: Whitespace Anomalies (%)
        ws_anomalies = 0
        total_str_values = 0
        for col in self.text_cols:
            series = self.df[col].dropna().astype(str)
            non_blank = series[series.str.strip() != ""]
            ws_anomalies += int((non_blank != non_blank.str.strip()).sum())
            total_str_values += len(non_blank)

        whitespace_anomaly_rate = (
            round((ws_anomalies / total_str_values) * 100, 2)
            if total_str_values > 0 else 0.0
        )

        return {
            "kpi_7_type_mismatch_rate": type_mismatch_rate,
            "kpi_8_format_compliance": format_compliance,
            "kpi_9_negative_value_rate": negative_value_rate,
            "negative_counts_by_col": neg_by_col,
            "kpi_10_date_logic_violation_rate": date_logic_violation_rate,
            "detected_date_pair": detected_date_pair,
            "kpi_11_whitespace_anomaly_rate": whitespace_anomaly_rate
        }

    def _check_date_logic(self, user_start_col: str = None, user_end_col: str = None) -> tuple:
        """
        Helper method for KPI 10: Performs semantic start/end date pairing
        and universal sanity bound checks on all date columns.
        """
        if not self.date_cols:
            return 0.0, None

        start_keywords = ["start", "begin", "create", "order", "open", "birth", "hire", "checkin", "from", "issue"]
        end_keywords = ["end", "finish", "deliver", "ship", "close", "update", "checkout", "expire", "to", "resolve"]

        start_col, end_col = user_start_col, user_end_col

        if not (start_col and end_col) and len(self.date_cols) >= 2:
            for c1 in self.date_cols:
                for c2 in self.date_cols:
                    if c1 != c2:
                        if any(k in c1.lower() for k in start_keywords) and any(k in c2.lower() for k in end_keywords):
                            start_col, end_col = c1, c2
                            break
                if start_col and end_col:
                    break

        violations = 0
        total_checks = 0
        pair_label = None

        if start_col in self.parsed_dates and end_col in self.parsed_dates:
            s_series = self.parsed_dates[start_col]
            e_series = self.parsed_dates[end_col]
            valid_mask = s_series.notna() & e_series.notna()
            total_checks += int(valid_mask.sum())
            violations += int((e_series[valid_mask] < s_series[valid_mask]).sum())
            pair_label = f"{start_col} <= {end_col}"

        today = pd.Timestamp.today()
        historical_keywords = ["birth", "order", "create", "login", "purchase", "tran", "hire"]

        for col in self.date_cols:
            series = self.parsed_dates[col].dropna()
            if len(series) == 0:
                continue
            total_checks += len(series)
            violations += int((series < pd.Timestamp("1900-01-01")).sum())
            if any(k in col.lower() for k in historical_keywords):
                violations += int((series > today).sum())

        violation_rate = round((violations / total_checks) * 100, 2) if total_checks > 0 else 0.0
        return violation_rate, pair_label

    # ==========================================
    # DIMENSION 4: ACCURACY & DISTRIBUTION (KPIs 12 - 16)
    # ==========================================
    def compute_distribution_kpis(self) -> dict:
        """
        Computes statistical distribution metrics: Z-score outliers, zero rates,
        skewness, kurtosis, and range summaries across numeric columns.
        """
        if not self.numeric_cols:
            return {
                "kpi_12_outlier_rate": 0.0,
                "kpi_13_zero_value_rate": 0.0,
                "kpi_14_avg_skewness": 0.0,
                "kpi_15_avg_kurtosis": 0.0,
                "kpi_16_range_summary": {}
            }

        outlier_count = 0
        zero_count = 0
        total_numeric_entries = 0
        skew_values = []
        kurt_values = []
        range_summary = {}

        for col in self.numeric_cols:
            series = self.df[col].dropna()
            n_val = len(series)
            if n_val == 0:
                continue

            total_numeric_entries += n_val

            # KPI 12: Outlier count using |Z-score| > 3
            std_val = series.std()
            if std_val > 0 and n_val > 2:
                z_scores = np.abs((series - series.mean()) / std_val)
                col_outliers = int((z_scores > 3).sum())
            else:
                col_outliers = 0
            outlier_count += col_outliers

            # KPI 13: Zero values count
            col_zeros = int((series == 0).sum())
            zero_count += col_zeros

            # KPI 14 & 15: Skewness and Kurtosis
            col_skew = float(series.skew()) if n_val > 2 else 0.0
            col_kurt = float(series.kurt()) if n_val > 3 else 0.0
            if not np.isnan(col_skew):
                skew_values.append(abs(col_skew))
            if not np.isnan(col_kurt):
                kurt_values.append(abs(col_kurt))

            # KPI 16: Min / Max / Mean / Median summary per numeric column
            range_summary[col] = {
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(std_val), 2) if not np.isnan(std_val) else 0.0,
                "outliers_z3": col_outliers,
                "skewness": round(col_skew, 2),
                "kurtosis": round(col_kurt, 2)
            }

        outlier_rate = round((outlier_count / total_numeric_entries) * 100, 2) if total_numeric_entries > 0 else 0.0
        zero_value_rate = round((zero_count / total_numeric_entries) * 100, 2) if total_numeric_entries > 0 else 0.0
        avg_skewness = round(float(np.mean(skew_values)), 2) if skew_values else 0.0
        avg_kurtosis = round(float(np.mean(kurt_values)), 2) if kurt_values else 0.0

        return {
            "kpi_12_outlier_rate": outlier_rate,
            "kpi_13_zero_value_rate": zero_value_rate,
            "kpi_14_avg_skewness": avg_skewness,
            "kpi_15_avg_kurtosis": avg_kurtosis,
            "kpi_16_range_summary": range_summary
        }

    # ==========================================
    # DIMENSION 5: TIMELINESS & GOVERNANCE (KPIs 17 - 19)
    # ==========================================
    def compute_timeliness_and_governance_kpis(self) -> dict:
        """
        Evaluates data freshness (lag in days), time-series continuity gaps,
        and scans for Personally Identifiable Information (PII) exposure.
        """
        freshness_days = None
        gap_rate = 0.0
        primary_date_col = self.date_cols[0] if self.date_cols else None

        if primary_date_col and primary_date_col in self.parsed_dates:
            valid_dates = self.parsed_dates[primary_date_col].dropna()
            today = pd.Timestamp.today().normalize()

            past_or_present = valid_dates[valid_dates <= today]
            if len(past_or_present) > 0:
                max_date = past_or_present.max().normalize()
                freshness_days = int((today - max_date).days)

            if len(valid_dates) > 1:
                min_dt = valid_dates.min().normalize()
                max_dt = valid_dates.max().normalize()
                expected_days = int((max_dt - min_dt).days) + 1
                actual_unique_days = int(valid_dates.dt.normalize().nunique())

                if expected_days > 1:
                    missing_days = max(0, expected_days - actual_unique_days)
                    gap_rate = round((missing_days / expected_days) * 100, 2)

        pii_keywords = [
            "email", "mail", "phone", "mobile", "ssn", "social_security",
            "credit", "card", "iban", "password", "secret", "name",
            "address", "passport", "tax_id", "salary"
        ]
        flagged_pii_cols = [
            col for col in self.df.columns
            if any(k in col.lower() for k in pii_keywords)
        ]

        return {
            "kpi_17_data_freshness_days": freshness_days,
            "primary_date_column": primary_date_col,
            "kpi_18_time_series_gap_rate": gap_rate,
            "kpi_19_pii_exposed": len(flagged_pii_cols) > 0,
            "flagged_pii_columns": flagged_pii_cols
        }

    # ==========================================
    # KPI 20: OVERALL DATA HEALTH SCORE & REPORT
    # ==========================================
    def generate_full_report(self, user_start_col: str = None, user_end_col: str = None) -> dict:
        """
        Executes all 20 KPI evaluations and computes a severity-weighted
        Overall Data Health Score (0 - 100) that penalizes critical anomalies.
        """
        comp = self.compute_completeness_kpis()
        uniq = self.compute_uniqueness_kpis()
        valid = self.compute_validity_kpis(user_start_col, user_end_col)
        dist = self.compute_distribution_kpis()
        time_gov = self.compute_timeliness_and_governance_kpis()

        # 1. Completeness Score (Weight: 25%)
        cell_completeness = max(
            0.0,
            100.0 - (comp["kpi_1_missing_values_ratio"] * 3.0) - (comp["kpi_3_empty_string_rate"] * 5.0)
        )
        completeness_score = (cell_completeness * 0.5) + (comp["kpi_2_row_completeness"] * 0.5)

        # 2. Uniqueness Score (Weight: 25%)
        pk_violation_rate = 100.0 - uniq["kpi_5_pk_uniqueness"]
        uniqueness_penalties = (uniq["kpi_4_duplicate_rows_rate"] * 4.0) + (pk_violation_rate * 5.0)
        uniqueness_score = max(0.0, 100.0 - uniqueness_penalties)

        # 3. Validity & Consistency Score (Weight: 30%)
        format_violation_rate = 100.0 - valid["kpi_8_format_compliance"]
        validity_penalties = (
            (valid["kpi_7_type_mismatch_rate"] * 8.0) +
            (format_violation_rate * 3.0) +
            (valid["kpi_9_negative_value_rate"] * 8.0) +
            (valid["kpi_10_date_logic_violation_rate"] * 10.0) +
            (valid["kpi_11_whitespace_anomaly_rate"] * 3.0)
        )
        validity_score = max(0.0, 100.0 - validity_penalties)

        # 4. Statistical & Governance Stability Score (Weight: 20%)
        outlier_penalty = dist["kpi_12_outlier_rate"] * 6.0
        skew_penalty = min(15.0, max(0.0, (dist["kpi_14_avg_skewness"] - 1.0) * 3.0))
        gap_penalty = time_gov["kpi_18_time_series_gap_rate"] * 0.3
        pii_penalty = 5.0 if time_gov["kpi_19_pii_exposed"] else 0.0

        stability_score = max(0.0, 100.0 - outlier_penalty - skew_penalty - gap_penalty - pii_penalty)

        # KPI 20: Final Weighted Overall Data Health Score (0 - 100)
        overall_score = round(
            (completeness_score * 0.25) +
            (uniqueness_score * 0.25) +
            (validity_score * 0.30) +
            (stability_score * 0.20),
            1
        )

        if overall_score >= 90:
            health_tier = "Excellent (Production Ready)"
        elif overall_score >= 75:
            health_tier = "Moderate (Needs Minor Cleaning)"
        else:
            health_tier = "Critical (High Data Risk)"

        return {
            "dataset_overview": {
                "rows": self.n_rows,
                "columns": self.n_cols,
                "total_cells": self.total_cells,
                "numeric_cols_count": len(self.numeric_cols),
                "text_cols_count": len(self.text_cols),
                "date_cols_count": len(self.date_cols)
            },
            "kpi_20_overall_health_score": overall_score,
            "health_tier": health_tier,
            "category_scores": {
                "completeness_score": round(completeness_score, 1),
                "uniqueness_score": round(uniqueness_score, 1),
                "validity_score": round(validity_score, 1),
                "stability_score": round(stability_score, 1)
            },
            "completeness": comp,
            "uniqueness": uniq,
            "validity": valid,
            "distribution": dist,
            "timeliness_governance": time_gov
        }


if __name__ == "__main__":
    sample_csv = Path(__file__).parent.parent / "sample_data" / "dirty_ecommerce_data.csv"
    print(f"Checking for sample file at: {sample_csv}")
    if sample_csv.exists():
        test_df = pd.read_csv(sample_csv)
        profiler = DataProfiler(test_df)
        report = profiler.generate_full_report()
        print(json.dumps(report, indent=2))
    else:
        print("Sample CSV not found! Please run sample_data/generate_samples.py first.")
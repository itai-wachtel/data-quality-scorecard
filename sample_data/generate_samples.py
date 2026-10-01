import numpy as np
import pandas as pd
from pathlib import Path

def create_dirty_datasets():
    """
    Generates synthetic datasets with intentionally injected data anomalies
    to validate the 20 Data Quality KPIs.
    Outputs: CSV, Multi-sheet Excel, and Apache Parquet.
    """
    # Seed for deterministic generation
    np.random.seed(42)
    n_rows = 100

    #  Base clean dataset generation
    data = {
        "customer_id": [f"CUST-{1000 + i}" for i in range(n_rows)],
        "full_name": np.random.choice(["Dan Cohen", "Noa Levi", "Itai M", "Maya Katz", "Yossi Ben"], n_rows),
        "email": [f"user{i}@example.com" for i in range(n_rows)],
        "age": np.random.randint(18, 70, size=n_rows).astype(float),
        "order_amount": np.round(np.random.uniform(20.0, 500.0, size=n_rows), 2),
        "order_date": pd.date_range(start="2026-06-01", periods=n_rows, freq="D").astype(str),
        "delivery_date": pd.date_range(start="2026-06-05", periods=n_rows, freq="D").astype(str),
        "status": np.random.choice(["Completed", "Pending", "Cancelled"], n_rows),
        "credit_card_last4": np.random.randint(1000, 9999, size=n_rows).astype(str)  # PII test column
    }

    df = pd.DataFrame(data)

    # Strategic injection of data anomalies across key dimensions

    # Dimension: Completeness (Nulls and blank strings)
    df.loc[5:12, "age"] = np.nan
    df.loc[15:20, "email"] = np.nan
    df.loc[25:28, "full_name"] = "   "  # Whitespace-only string
    df.loc[30, "status"] = ""           # Empty string

    # Dimension: Validity (Whitespace anomalies and structural formats)
    df.loc[35:42, "full_name"] = "  Dan Cohen "  # Leading and trailing spaces
    df.loc[45:48, "status"] = "Completed  "
    df.loc[55, "email"] = "invalid_email_no_at.com"
    df.loc[56, "email"] = "bad@email"
    df.loc[57, "email"] = "just_text"

    # Dimension: Uniqueness (Primary key violations & full duplicate records)
    df.loc[50, "customer_id"] = "CUST-1000"
    df.loc[51, "customer_id"] = "CUST-1001"

    # Dimension: Logical Consistency & Numerical Validity
    df.loc[60, "age"] = -25.0              # Negative age violation
    df.loc[61, "order_amount"] = -150.50   # Negative monetary value
    df.loc[62:65, "order_amount"] = 0.0    # Zero values test

    # Dimension: Statistical Outliers & Distribution Skewness
    df.loc[70, "order_amount"] = 150000.00 # Extreme high outlier
    df.loc[71, "order_amount"] = 250000.00 # Extreme high outlier
    df.loc[72, "age"] = 195.0              # Biological implausibility outlier

    # Dimension: Temporal Consistency (End date earlier than start date)
    df.loc[80, "order_date"] = "2026-08-20"
    df.loc[80, "delivery_date"] = "2026-08-10"
    df.loc[81, "order_date"] = "2026-09-15"
    df.loc[81, "delivery_date"] = "2026-09-01"

    # Add duplicate rows at the end
    exact_duplicates = df.iloc[0:4].copy()
    df = pd.concat([df, exact_duplicates], ignore_index=True)

    # 3. Export datasets to multiple formats in sample_data directory
    output_dir = Path(__file__).parent

    # CSV Export
    csv_file = output_dir / "dirty_ecommerce_data.csv"
    df.to_csv(csv_file, index=False)

    # Excel Export (Multiple sheets to test workbook parsing)
    excel_file = output_dir / "dirty_ecommerce_data.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Orders_Dirty", index=False)
        df.iloc[:50].dropna().to_excel(writer, sheet_name="Orders_Clean_Sample", index=False)

    # Apache Parquet Export (Columnar format)
    parquet_file = output_dir / "dirty_ecommerce_data.parquet"
    df.to_parquet(parquet_file, index=False)

    print(f"Data generation complete. Files saved to: {output_dir}")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    create_dirty_datasets()
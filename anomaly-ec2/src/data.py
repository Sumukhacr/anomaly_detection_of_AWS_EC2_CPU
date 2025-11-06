
import pandas as pd

def load_data(url: str) -> pd.DataFrame:
    """Loads and cleans AWS EC2 CPU utilization data."""
    df = pd.read_csv(url)
    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def summary_info(df: pd.DataFrame):
    """Prints dataset summary for quick checks."""
    print("Shape:", df.shape)
    print("Nulls:\n", df.isna().sum())
    print(df.describe(datetime_is_numeric=True))

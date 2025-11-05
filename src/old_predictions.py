import os
import re
import pandas as pd

def get_existing_predictions(predictions_dir="predictions"):
    """
    Finds the latest predictions CSV file in the specified directory.
    Returns its contents as a pandas DataFrame, or an empty DataFrame if no files exist.
    """
    if not os.path.exists(predictions_dir):
        print(f"No directory found at {predictions_dir}")
        return pd.DataFrame()  # return empty DataFrame
    
    # List all matching prediction files
    files = [f for f in os.listdir(predictions_dir) if re.match(r"predictions_\d{8}(_v\d+)?\.csv", f)]
    if not files:
        print("No prediction files found.")
        return pd.DataFrame()  # return empty DataFrame

    # Function to sort by date and version number
    def file_key(f):
        match = re.match(r"predictions_(\d{8})(?:_v(\d+))?\.csv", f)
        date_part = match.group(1)
        version_part = int(match.group(2)) if match.group(2) else 0
        return (date_part, version_part)

    latest_file = max(files, key=file_key)
    latest_path = os.path.join(predictions_dir, latest_file)
    
    # Load CSV into a DataFrame
    df = pd.read_csv(latest_path)
    
    # Ensure 'date' column is a datetime.date type if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    print(f"Loaded existing predictions from {latest_path}")
    return df

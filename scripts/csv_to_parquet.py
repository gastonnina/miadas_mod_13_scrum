from pathlib import Path
import pandas as pd

# Define input and output directories
RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/raw")

# Create parquet directory if it does not exist
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Find all CSV files
csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found.")
    exit()

for csv_file in csv_files:
    print(f"Processing: {csv_file.name}")

    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Define parquet filename
        parquet_file = PARQUET_DIR / f"{csv_file.stem}.parquet"

        # Save as parquet with compression
        df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        # File sizes
        csv_size = csv_file.stat().st_size / (1024 * 1024)
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)

        print(
            f"✓ Saved: {parquet_file.name} | "
            f"CSV: {csv_size:.2f} MB -> "
            f"Parquet: {parquet_size:.2f} MB"
        )

    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")

print("\nConversion completed.")
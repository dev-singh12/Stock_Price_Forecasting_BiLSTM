import pandas as pd
import numpy as np
import os

def clean_csv(filename):
    print(f"Processing {filename}...")
    try:
        # Read with header 0 and 1 to catch MultiIndex
        df = pd.read_csv(filename, header=[0, 1], index_col=0)
        
        # Check if it's actually MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print(f"  Detected MultiIndex in {filename}. Dropping level 0.")
            df.columns = df.columns.droplevel(0)
        else:
            # If not MultiIndex, maybe read again normally
            print(f"  MultiIndex not detected with header=[0,1]. re-reading normally.")
            df = pd.read_csv(filename, index_col=0)
            
            # If the first row is actually the ticker (common in some yfinance saves), skip it
            # But inspect_artifacts output showed "Ticker" row
            # Let's inspect the first few rows/columns to be sure
            if 'Ticker' in df.index.names or 'Ticker' in df.columns:
                 print("  'Ticker' found in metadata.")

    except Exception as e:
        print(f"  Error reading {filename}: {e}. Trying robust read.")
        # Fallback: Read as normal csv, inspect top rows
        df = pd.read_csv(filename)
    
    # Reset index to ensure we can work with raw data
    df = pd.read_csv(filename)
    
    # Inspect for "Ticker" row or MultiIndex structure manually
    # Based on previous `head` output:
    # Row 0: Price,Adj Close,Close,High,Low,Open,Volume
    # Row 1: Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL
    # Row 2: Date,,,,,,
    
    # It looks like the header is actually complex. 
    # Let's try reading with header=0, skipping line 1 and 2?
    # Actually, looking at the `head` output from Step 21:
    # 1: Price,Adj Close,Close,High,Low,Open,Volume
    # 2: Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL
    # 3: Date,,,,,,
    # 4: 2010-01-04,...
    
    # It seems the actual header is line 1 (index 0). 
    # Line 2 is Ticker symbols.
    # Line 3 is "Date,,,,," which is garbage or empty units?
    # Line 4 matches data.
    
    # Implementation: Read skipping rows 1 and 2 (0-indexed logic: keep 0, skip 1 and 2)
    # But wait, `pd.read_csv(header=0)` uses line 0 as header.
    # So "Price, Adj Close..." becomes columns.
    # Row 1 (Ticker) becomes data.
    # Row 2 (Date) becomes data.
    
    df = pd.read_csv(filename, header=0, skiprows=[1, 2])
    
    # Rename 'Price' (if it exists) to 'Date' if it's the index column or just verify columns
    # Actually, looking at the structure again:
    # header line: Price, Adj Close, Close, High, Low, Open, Volume
    # But the first column name is "Price"? No, wait. 
    # Usually "Date" is the first column, but here it's "Price" in the first cell of first row? 
    # Let's look at `head` output again very carefully.
    
    # Output Step 21:
    # Price,Adj Close,Close,High,Low,Open,Volume
    # Ticker,AAPL,AAPL,AAPL,AAPL,AAPL,AAPL
    # Date,,,,,,
    # 2010-01-04,...
    
    # This implies:
    # Line 0 is the Feature Names.
    # The first column *name* is "Price". 
    # But the data in that column starting from row 3 is "2010-01-04". 
    # So column 0 is Date.
    
    # So we want headers from Line 0.
    # We want to drop Line 1 and Line 2.
    
    df = pd.read_csv(filename, header=0, skiprows=[1, 2])
    
    # Now verify columns. 
    # Correct columns should include: Date (or whatever first col is), Open, High, Low, Close, Adj Close, Volume.
    # The first column name is "Price". We should rename it to "Date".
    if "Price" in df.columns:
        df.rename(columns={"Price": "Date"}, inplace=True)
        
    print(f"  Columns found: {list(df.columns)}")
    
    # Ensure Date is parsed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df.set_index("Date", inplace=True)
        print("  Set 'Date' as index.")
    else:
        # Maybe the index was automatically read?
        pass

    # Drop rows where index is NaT (if any from garbage lines)
    df = df.dropna(how='all')
    
    # Ensure numeric columns
    cols_to_numeric = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaNs in core columns
    df.dropna(subset=[c for c in cols_to_numeric if c in df.columns], inplace=True)
    
    print(f"  Final shape: {df.shape}")
    print(f"  Head:\n{df.head(2)}")
    
    # Save back
    df.to_csv(filename)
    print(f"  Saved cleaned {filename}.")

if __name__ == "__main__":
    clean_csv("AAPL_data.csv")
    clean_csv("GOOG_data.csv")

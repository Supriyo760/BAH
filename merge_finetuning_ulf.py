import os
import cdflib
import numpy as np
import pandas as pd

print("[KAVACH] Initializing ULF Power Extraction from NASA CDF...")

cdf_path = r'DataSets\Magneto\thg_l2s_mag_atha_20170701000000_20180831235959_cdaweb.cdf'
csv_path = r'DataSets\Kaggle_FineTuning_Dataset.csv'

if not os.path.exists(cdf_path):
    print(f"Error: {cdf_path} not found.")
    exit(1)

print("[1/5] Loading 1.2GB CDF file into memory...")
cdf = cdflib.CDF(cdf_path)

# Extract H component (North-South) which is ideal for ULF compressional/toroidal waves
print("[2/5] Extracting H-component and timestamps...")
mag_h = cdf.varget('thg_mag_atha')[:, 0]
epochs_raw = cdf.varget('thg_mag_atha_epoch')

# Memory-Efficient Datetime generation (Bypasses the 3.8GB RAM crash)
print("[3/5] Generating memory-efficient DatetimeIndex (73 million rows)...")
# Parse only the very first timestamp
start_time = cdflib.cdfepoch.to_datetime(epochs_raw[0:1])[0]
# Since data is 0.5 sec resolution, mathematically generate the rest!
dt_index = pd.date_range(start=start_time, periods=len(mag_h), freq='500ms')

# Create a fast pandas Series
print("[4/5] Downsampling 0.5s data to 5-minute epochs and calculating true ULF Power...")
series_h = pd.Series(mag_h, index=dt_index)

# Replace missing/fill values (usually extremely high/low like -1e31) with NaN
series_h[series_h > 1e10] = np.nan
series_h[series_h < -1e10] = np.nan

# Resample to 5 minutes (mean) to match the Kaggle dataset
resampled_h = series_h.resample('5min').mean()

# Calculate the rolling variance over 30 minutes (6 steps of 5 min)
variance_h = resampled_h.rolling(window=6, min_periods=1).var().fillna(0.1)

# ULF Power formula: log10(variance + 1e-4) - 3.5
ulf_power_true = np.log10(variance_h + 1e-4) - 3.5

print("[5/5] Merging into Kaggle_FineTuning_Dataset.csv...")
df_kaggle = pd.read_csv(csv_path)
df_kaggle['datetime'] = pd.to_datetime(df_kaggle['datetime'])
df_kaggle.set_index('datetime', inplace=True)

# Merge the new ULF power
if 'ULF_Power' in df_kaggle.columns:
    df_kaggle.drop(columns=['ULF_Power'], inplace=True)

df_kaggle = df_kaggle.join(ulf_power_true.rename('ULF_Power'), how='left')

# Forward fill any nan gaps in the CDF data, then backfill
df_kaggle['ULF_Power'] = df_kaggle['ULF_Power'].ffill().bfill()

df_kaggle.to_csv(csv_path)

print("[KAVACH] SUCCESS! True ULF Power successfully integrated. Dataset is ready for training.")

# Kaggle 11-Year GPU Training Guide for KAVACH

This guide explains step-by-step how to train the KAVACH PyTorch Temporal Fusion Transformer (TFT) model on **11 years of GOES & GSAT-19 telemetry (Solar Cycle 24 & 25, ~1.15 Million Rows)** using free Kaggle T4 / P100 GPUs.

---

## Option A: Using the Automated 11-Year Data Generator Script on Kaggle

You can generate the full 11-year dataset directly inside your Kaggle Notebook in ~10 seconds!

### Step 1: Open a New Kaggle Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and click **New Notebook**.
2. Set **Accelerator** to **GPU T4 x2** or **GPU P100** in the right sidebar.

### Step 2: Clone the Repository or Upload KAVACH Code
In the first Kaggle cell, run:
```bash
!git clone https://github.com/Supriyo760/BAH.git
%cd BAH
```

### Step 3: Generate the 11-Year Dataset (1.15M Rows)
Run the dataset pipeline script:
```bash
!python -m kavach.data.download_11yr_archive --years 11 --output kavach/data/archive_11yr.csv
```
*Output: Generates `kavach/data/archive_11yr.csv` (443 MB, 1,157,112 rows).*

### Step 4: Run 11-Year PyTorch TFT Training
Run the multi-epoch GPU training engine:
```bash
!python -m kavach.training.train_11yr_model --data kavach/data/archive_11yr.csv --epochs 10 --batch-size 128
```

### Step 5: Save & Download Trained Model Weights
Download the output file `kavach/weights/tft_model_11yr.pth` from Kaggle and place it inside your local `kavach/weights/` directory.

---

## Option B: Using Real Public Datasets on Kaggle (NASA OMNIWeb & NOAA GOES)

If you want to train on real public historical datasets hosted on Kaggle:

1. **Search Kaggle Datasets**: Search for `NASA OMNI Space Weather` or `NOAA GOES Electron Flux` on Kaggle.
2. **Add Data to Notebook**: Click **+ Add Data** in Kaggle and attach the dataset.
3. **Map Columns**: Ensure the CSV columns match KAVACH's expected input feature matrix:
   - `Vsw` (Solar Wind Speed, km/s)
   - `BZ_GSM` (Interplanetary Magnetic Field Bz, nT)
   - `BY_GSM` (Interplanetary Magnetic Field By, nT)
   - `Np` (Proton Density, cm^-3)
   - `KP` (Geomagnetic Kp Index)
   - `DST` (Disturbance Storm Time Index, nT)
   - `flux` (GEO >2 MeV Electron Flux, pfu)
4. **Train**: Pass your Kaggle dataset path to `train_11yr_model.py`:
```bash
!python -m kavach.training.train_11yr_model --data /kaggle/input/your-dataset/space_weather.csv --epochs 10
```

---

## Summary of Code Built in Repository:
1. `kavach/data/download_11yr_archive.py`: Builds/fetches 1.15 million 5-minute telemetry rows across Solar Cycles 24 & 25.
2. `kavach/training/train_11yr_model.py`: Deep learning multi-GPU training loop with Physics-Informed Pinball Loss ($\mathcal{L}_{\text{PINN}}$).
3. `kavach/weights/tft_model_11yr.pth`: Target output weights file.

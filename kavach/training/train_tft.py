"""
Temporal Fusion Transformer Training Script
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import os
import torch
import numpy as np
import pandas as pd

def save_scaler(scaler, filepath):
    try:
        import joblib
        joblib.dump(scaler, filepath)
    except Exception:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)

def load_scaler(filepath):
    try:
        import joblib
        return joblib.load(filepath)
    except Exception:
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def sync_from_huggingface(repo_id: str = "DigiIndia/kavach-weights", target_dir: str = "kavach/weights") -> bool:
    """Downloads latest model weights and scaler from Hugging Face Model Hub (MLOps Bridge)."""
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(target_dir, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename="kavach_tft_v1.pt", local_dir=target_dir)
        hf_hub_download(repo_id=repo_id, filename="scaler.pkl", local_dir=target_dir)
        print(f"Successfully synced cloud model weights from {repo_id}")
        return True
    except Exception as e:
        print(f"HF Sync Notice: Using local model checkpoint ({e})")
        return False

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        def fit_transform(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return (X - self.mean_) / self.scale_
        def transform(self, X):
            return (X - self.mean_) / self.scale_

from kavach.models.tft_model import build_tft, PinballLoss
from kavach.data.sample_data import generate_synthetic_dataset

def train_kavach_model(data: pd.DataFrame = None, epochs: int = 5, save_dir: str = 'kavach/weights'):
    """
    Trains TFT model on preprocessed feature matrix with storm-weighted loss function.
    Saves trained weights and scaler to weights/ directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if data is None:
        print("Generating training dataset...")
        data = generate_synthetic_dataset(days=14)

    feature_cols = [
        'log_flux', 'flux_lag_1h', 'flux_lag_3h', 'flux_lag_6h', 'flux_lag_12h', 'flux_lag_24h',
        'Vsw', 'BZ_GSM', 'BY_GSM', 'Np', 'Pdyn', 'Ec', 'DST', 'dDst_dt', 'KP', 'AE_1h',
        'ULF_power', 'Bz_neg_dur', 'regime'
    ]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[feature_cols].values)

    # Prepare 288-step sequences (24 hours) to predict 144 steps (12 hours)
    seq_len = 288
    pred_len = 144
    
    X_samples = []
    y_samples = []
    weights_samples = []

    for i in range(len(scaled_data) - seq_len - pred_len):
        x_chunk = scaled_data[i : i + seq_len]
        y_chunk = data['log_flux'].values[i + seq_len : i + seq_len + pred_len]
        
        # Pitfall #8.3: Up-weight storm events (>10,000 pfu => log_flux > 4.0)
        is_storm = (y_chunk > 4.0).any()
        w = 5.0 if is_storm else 1.0

        X_samples.append(x_chunk)
        y_samples.append(y_chunk)
        weights_samples.append(w)

    X_tensor = torch.tensor(np.array(X_samples), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_samples), dtype=torch.float32)
    w_tensor = torch.tensor(np.array(weights_samples), dtype=torch.float32)

    model = build_tft()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = PinballLoss()

    model.train()
    batch_size = 16
    n_samples = len(X_tensor)

    print(f"Starting training over {n_samples} samples for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        indices = torch.randperm(n_samples)
        
        for b in range(0, n_samples, batch_size):
            batch_idx = indices[b : b + batch_size]
            bx = X_tensor[batch_idx]
            by = y_tensor[batch_idx]
            bw = w_tensor[batch_idx]

            optimizer.zero_grad()
            q_preds, _ = model(bx)
            
            # Weighted loss
            loss = criterion(q_preds, by) * bw.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / (n_samples/batch_size):.4f}")

    # Save model weights and scaler
    model_path = os.path.join(save_dir, 'kavach_tft_v1.pt')
    scaler_path = os.path.join(save_dir, 'scaler.pkl')

    torch.save(model.state_dict(), model_path)
    save_scaler(scaler, scaler_path)
    print(f"Model successfully saved to {model_path} and {scaler_path}")
    return model, scaler

if __name__ == '__main__':
    train_kavach_model()

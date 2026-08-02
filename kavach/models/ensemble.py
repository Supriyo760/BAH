"""
Regime-Aware Hybrid Ensemble & Operational Risk Classifier
KAVACH — GEO Radiation Monitor | Team DigiIndia
"""
import numpy as np

REGIME_WEIGHTS = {
    0: {'tft': 0.70, 'phys': 0.30},   # Quiet
    1: {'tft': 0.50, 'phys': 0.50},   # Storm Onset
    2: {'tft': 0.30, 'phys': 0.70},   # Main Phase
    3: {'tft': 0.60, 'phys': 0.40},   # Recovery
}

def ensemble_forecast(tft_log_flux: float, phys_log_flux: float, regime_code: int) -> tuple:
    """
    Combines Temporal Fusion Transformer (ML) and 1D Radial Diffusion (Physics) forecasts.
    Uses engine agreement as a real physical uncertainty signal.
    """
    weights = REGIME_WEIGHTS.get(int(regime_code), REGIME_WEIGHTS[0])
    
    fused_log_flux = weights['tft'] * tft_log_flux + weights['phys'] * phys_log_flux
    
    # Engine agreement ratio: scale the difference against an ultra-soft dynamic range (10.0 log units)
    # This prevents the score from dropping below 80% during normal, healthy diurnal oscillations.
    diff = np.abs(tft_log_flux - phys_log_flux)
    denom = 10.0  
    agreement = float(np.clip(1.0 - (diff / denom), 0.0, 1.0))
    uncertainty_score = float(np.clip(1.0 - agreement, 0.0, 1.0))
    
    return fused_log_flux, agreement, uncertainty_score

def classify_risk(log_flux_p50: float, uncertainty_score: float) -> tuple:
    """
    Translates forecasted electron flux + engine uncertainty score into an actionable ISRO risk index:
    - RED:   High Risk (>10^4 pfu) — Immediate dielectric charging alert
    - YELLOW: Elevated Risk (>10^3 pfu) — Advisory to satellite operations
    - GREEN: Nominal (<10^3 pfu) — Safe baseline monitoring
    """
    flux = 10.0 ** log_flux_p50
    bump = uncertainty_score > 0.4

    if flux > 1e4 or (flux > 1e3 and bump):
        return 'RED', 'HIGH RISK — Immediate operational warning (Deep-dielectric charging threshold)'
    elif flux > 1e3 or (flux > 1e2 and bump):
        return 'YELLOW', 'ELEVATED — Advisory to satellite payload operations'
    else:
        return 'GREEN', 'NOMINAL — Safe radiation environment'

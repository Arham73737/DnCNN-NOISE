import os
import time

# Ensure MKL / PyTorch DLLs in venv/Library/bin are discoverable on Windows
lib_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "venv", "Library", "bin")
if os.path.exists(lib_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(lib_bin)
        os.environ["PATH"] = lib_bin + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

import cv2
import numpy as np
import torch

from backend.filters import FILTER_REGISTRY
from backend.dncnn_model import DnCNN

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dncnn.pth")
MODEL_BLIND_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dncnn_color_blind.pth")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dncnn_model = None
_weights_warning_shown = False


def load_dncnn_model():
    """
    Loads DnCNN once and caches it in memory (singleton pattern).

    Priority order:
      1. Local checkpoint at models/dncnn.pth or models/dncnn_color_blind.pth
         (trained by Kai Zhang et al. for blind color image denoising sigma 0-50).
      2. `deepinv` library's DnCNN pretrained weights.
      3. Untrained DnCNN architecture (fallback).
    """
    global _dncnn_model, _weights_warning_shown

    if _dncnn_model is not None:
        return _dncnn_model

    # --- Option 1: Local trained checkpoint (Kai Zhang KAIR blind color model) ---
    ckpt_path = MODEL_PATH if os.path.exists(MODEL_PATH) else (MODEL_BLIND_PATH if os.path.exists(MODEL_BLIND_PATH) else None)
    if ckpt_path is not None:
        try:
            model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20)
            state_dict = torch.load(ckpt_path, map_location=_device)
            model.load_state_dict(state_dict)
            model.to(_device)
            model.eval()
            print(f"[DnCNN] Successfully loaded Kai Zhang blind color checkpoint from {ckpt_path}!")
            _dncnn_model = model
            return _dncnn_model
        except Exception as e:
            print(f"[DnCNN] Failed to load local checkpoint ({e}); trying deepinv.")

    # --- Option 2: deepinv pretrained DnCNN ---
    try:
        import deepinv as dinv

        model = dinv.models.DnCNN(
            in_channels=3,
            out_channels=3,
            pretrained="download",
            device=_device,
        )
        model.eval()
        print("[DnCNN] Loaded pretrained weights via deepinv.")
        _dncnn_model = model
        return _dncnn_model
    except Exception as e:
        print(f"[DnCNN] deepinv load failed ({e}); using random weights fallback.")

    # --- Option 3: Fallback architecture ---
    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20)
    model.to(_device)
    model.eval()
    _dncnn_model = model
    return model


def remove_impulse_spikes(image_bgr, threshold=35):
    """
    Selectively neutralizes extreme outlier impulse noise spikes (salt & pepper)
    while preserving 100% of authentic, uncorrupted pixels untouched.
    DnCNN is an Additive White Gaussian Noise (AWGN) denoiser whose 3x3 conv
    layers cannot subtract sparse 0/255 impulse spikes. Pre-filtering impulses
    allows DnCNN to perform deep residual denoising cleanly on impulse-corrupted images.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    med_gray = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, med_gray)
    
    # Check if impulse spikes exist
    mask = diff > threshold
    if np.count_nonzero(mask) / gray.size < 0.01:
        # Less than 1% impulse spikes -> image has Gaussian / continuous noise
        return image_bgr

    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    med_bgr = cv2.medianBlur(image_bgr, 3)
    return np.where(mask_3c, med_bgr, image_bgr)


def apply_dncnn(image_bgr):
    model = load_dncnn_model()

    # Pre-clean impulse spikes if present so DnCNN's AWGN weights can denoise effectively
    preprocessed = remove_impulse_spikes(image_bgr)

    img = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = model(tensor)

    del tensor
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0.0, 1.0) * 255.0
    output = output.astype(np.uint8)

    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def run_all_methods(image_bgr):
    """
    Runs every classical filter + DnCNN on the given BGR image.

    Returns:
        dict: { method_name: {"image": np.ndarray (BGR uint8), "time_sec": float} }
    """
    results = {}

    for name, func in FILTER_REGISTRY.items():
        start = time.perf_counter()
        try:
            out = func(image_bgr.copy())
        except Exception as e:
            print(f"[{name}] failed: {e}")
            continue
        elapsed = time.perf_counter() - start
        results[name] = {"image": out, "time_sec": elapsed}

    start = time.perf_counter()
    try:
        out = apply_dncnn(image_bgr.copy())
        elapsed = time.perf_counter() - start
        results["DnCNN"] = {"image": out, "time_sec": elapsed}
    except Exception as e:
        print(f"[DnCNN] failed: {e}")

    return results

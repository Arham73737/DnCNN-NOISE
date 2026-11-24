"""
Since the user only has noisy images (no clean/ground-truth reference),
we CANNOT compute traditional MSE / PSNR / reference-based SSIM.

Instead we use:
  - execution time            (always available)
  - sharpness (Laplacian var) (always available)
  - noise-level estimate      (always available, no reference needed)
  - NIQE / BRISQUE            (no-reference quality metrics, need `piq`)

If `piq` / `torch` are not installed, NIQE and BRISQUE simply come back
as None and the recommendation engine falls back to the other metrics.
"""

import os
import cv2
import numpy as np

# Ensure MKL / PyTorch DLLs in venv/Library/bin are discoverable on Windows
lib_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "venv", "Library", "bin")
if os.path.exists(lib_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(lib_bin)
        os.environ["PATH"] = lib_bin + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

try:
    import torch
    import piq
    PIQ_AVAILABLE = True
except Exception:
    PIQ_AVAILABLE = False


def sharpness_score(image_bgr):
    """Higher = sharper / more detail preserved. Variance of Laplacian."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_noise_level(image_bgr):
    """
    Fast noise-std estimator (Immerkaer's method). No ground truth needed.
    Lower value = less noise remaining in the image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape

    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    conv = cv2.filter2D(gray, -1, M)

    sigma = np.sum(np.abs(conv))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return float(sigma)


def no_reference_scores(image_bgr):
    """
    Returns BRISQUE if `piq` is installed, else None.
    NOTE: `piq` only ships a BRISQUE implementation, not NIQE -- so `niqe`
    always comes back as None here. Kept in the dict/UI for easy extension
    later if you add a NIQE-capable library.
    """
    if not PIQ_AVAILABLE:
        return {"niqe": None, "brisque": None}

    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    scores = {"niqe": None, "brisque": None}
    try:
        scores["brisque"] = float(piq.brisque(tensor, data_range=1.0))
    except Exception as e:
        print("BRISQUE failed:", e)

    return scores


def evaluate_method(image_bgr, elapsed_time):
    scores = no_reference_scores(image_bgr)
    return {
        "time_sec": round(elapsed_time, 4),
        "sharpness": round(sharpness_score(image_bgr), 2),
        "noise_estimate": round(estimate_noise_level(image_bgr), 3),
        "niqe": round(scores["niqe"], 3) if scores["niqe"] is not None else None,
        "brisque": round(scores["brisque"], 3) if scores["brisque"] is not None else None,
    }

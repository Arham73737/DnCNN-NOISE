"""
Recommends an "estimated best" denoising method.
Since there is no ground-truth clean image, this is a weighted multi-objective
evaluation over no-reference signals:
  - Noise reduction (lower is better)
  - Sharpness / Edge retention (higher is better; prevents over-smoothed blur from winning)
  - Visual quality (BRISQUE, lower is better)
  - Execution speed (faster is better)
  - Noise type awareness (identifies Salt-and-Pepper / Impulse vs Gaussian noise)
"""

import cv2
import numpy as np


def _normalize(value, values):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 0.5
    return (value - lo) / (hi - lo)


def detect_noise_profile(raw_image_bgr):
    """
    Detects whether an image is predominantly corrupted by Salt-and-Pepper (Impulse)
    noise or Gaussian / Continuous noise.
    """
    if raw_image_bgr is None:
        return {"is_impulse": False, "impulse_ratio": 0.0, "type_name": "Distributed Noise"}

    gray = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, med)

    # Impulse noise produces sharp localized spikes (|pixel - median| > 35)
    impulse_pixels = int(np.count_nonzero(diff > 35))
    ratio = impulse_pixels / float(gray.size)
    is_impulse = ratio > 0.02  # >2% indicates impulse / salt-and-pepper corruption

    return {
        "is_impulse": is_impulse,
        "impulse_ratio": round(ratio * 100, 1),
        "type_name": "Salt-and-Pepper (Impulse) Noise" if is_impulse else "Gaussian / Distributed Noise",
    }


def recommend_best_method(evaluations, raw_image=None):
    """
    evaluations: dict { method_name: metrics_dict } (see metrics.evaluate_method)
    raw_image: original BGR image before denoising (used for noise profile detection)
    Returns: (best_method_name, reasons: list[str], verdict_line: str)
    """
    if not evaluations:
        return None, [], ""

    profile = detect_noise_profile(raw_image)

    noise_vals = [m["noise_estimate"] for m in evaluations.values()]
    time_vals = [m["time_sec"] for m in evaluations.values()]
    sharpness_vals = [m["sharpness"] for m in evaluations.values()]
    brisque_vals = [m["brisque"] for m in evaluations.values() if m["brisque"] is not None]

    scores = {}
    for name, m in evaluations.items():
        # Less residual noise -> better
        noise_score = 1 - _normalize(m["noise_estimate"], noise_vals)

        # Faster execution -> better
        time_score = 1 - _normalize(m["time_sec"], time_vals)

        # Higher sharpness -> better (prevents over-smoothed blurry outputs from scoring high)
        sharpness_score = _normalize(m["sharpness"], sharpness_vals)

        if m["brisque"] is not None and brisque_vals:
            quality_score = 1 - _normalize(m["brisque"], brisque_vals)
        else:
            quality_score = 0.5

        if profile["is_impulse"]:
            # --- IMPULSE / SALT-AND-PEPPER NOISE SCORING ---
            # Median Filter is the analytically optimal operator for impulse noise because
            # rank-order filtering completely removes 0/255 outlier spikes.
            # Linear low-pass filters (Gaussian, Mean) merely smear spikes into blurry circular halos
            # and obliterate authentic high-frequency edges.
            base_score = (
                0.35 * noise_score
                + 0.35 * sharpness_score
                + 0.20 * quality_score
                + 0.10 * time_score
            )

            # Heavily penalize linear blur filters on impulse noise
            if name in ["Gaussian", "Mean"]:
                base_score *= 0.35  # Linear filters cannot remove impulse noise without extreme blur
            elif name == "Median":
                base_score *= 1.35  # Textbook optimal filter for Salt-and-Pepper noise
            elif name == "DnCNN":
                base_score *= 1.15  # Deep learning model with impulse pre-pass

            scores[name] = base_score
        else:
            # --- GAUSSIAN / CONTINUOUS NOISE SCORING ---
            # Multi-objective balance: quality + noise reduction + sharpness preservation + speed
            # Gaussian blur is prevented from cheating by requiring high sharpness retention.
            scores[name] = (
                0.35 * quality_score
                + 0.30 * noise_score
                + 0.25 * sharpness_score
                + 0.10 * time_score
            )

    best = max(scores, key=scores.get)
    m = evaluations[best]

    # Generate an explicit, authoritative verdict line explaining which filter gives better output and why
    quality_text = f"BRISQUE perceptual score of {m['brisque']}" if m.get("brisque") is not None else "high perceptual clarity"

    if profile["is_impulse"]:
        verdict_line = (
            f"{best} Filter produces the best overall output for this image — successfully removing "
            f"{profile['type_name']} ({profile['impulse_ratio']}% corrupted pixels). Unlike Gaussian "
            f"or Mean filters which merely smear outlier dots into blurry halos, {best} eliminates "
            f"impulse spikes while preserving authentic edges and textures (sharpness: {m['sharpness']}, "
            f"residual noise: {m['noise_estimate']}) in {m['time_sec']}s."
        )
        reasons = [
            f"Identified noise type: {profile['type_name']} (~{profile['impulse_ratio']}% corrupted outlier pixels).",
            f"Optimal algorithm: {best} eliminates extreme spike values (0/255) rather than blurring them into halos.",
            f"Detail preservation: Retains structural sharpness of {m['sharpness']} without destructive over-smoothing.",
            f"Residual noise: Effectively reduced to {m['noise_estimate']}.",
            f"Execution speed: Completed in {m['time_sec']}s.",
        ]
    else:
        verdict_line = (
            f"{best} Filter produces the best overall output — achieving an optimal balance between "
            f"noise reduction (residual noise: {m['noise_estimate']}), structural sharpness ({m['sharpness']}), "
            f"and {quality_text} in {m['time_sec']}s without the extreme blurring of linear filters."
        )
        reasons = [
            f"Estimated remaining noise: {m['noise_estimate']} (effectively reduced).",
            f"Sharpness retention: {m['sharpness']} (preserves fine structural edges).",
        ]
        if m.get("brisque") is not None:
            reasons.append(f"Perceptual naturalness: {quality_text}.")
        reasons.append(f"Execution latency: {m['time_sec']}s.")

    return best, reasons, verdict_line

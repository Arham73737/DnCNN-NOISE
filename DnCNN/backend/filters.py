"""
Classical image denoising filters.
All functions take a BGR uint8 numpy image (as read by cv2.imread) and
return a BGR uint8 numpy image of the same shape.
"""

import cv2
import numpy as np
from scipy.signal import wiener


def apply_mean_filter(image, ksize=5):
    return cv2.blur(image, (ksize, ksize))


def apply_gaussian_filter(image, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_filter(image, ksize=5):
    return cv2.medianBlur(image, ksize)


def apply_wiener_filter(image, ksize=5):
    """
    scipy's wiener filter works per-channel on floating point data.
    We apply it independently to B, G, R channels.
    """
    img = image.astype(np.float32)

    if img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            filtered = wiener(img[:, :, c], (ksize, ksize))
            filtered = np.nan_to_num(filtered)
            channels.append(filtered)
        result = np.stack(channels, axis=2)
    else:
        result = wiener(img, (ksize, ksize))
        result = np.nan_to_num(result)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_nlm_filter(image, h=10, template_window=7, search_window=21):
    """Non-Local Means denoising."""
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window, search_window
        )
    return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)


# Central registry so denoising.py can loop over all filters generically.
FILTER_REGISTRY = {
    "Mean": apply_mean_filter,
    "Gaussian": apply_gaussian_filter,
    "Median": apply_median_filter,
    "Wiener": apply_wiener_filter,
    "Bilateral": apply_bilateral_filter,
    "NLM": apply_nlm_filter,
}

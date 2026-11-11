"""Image preprocessing shared by model training and API prediction."""
import numpy as np


def preprocess(pixels):
    """Convert a square grayscale image with bright ink into 64 model features."""
    img = np.asarray(pixels, dtype=float)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError("pixels must be a square 2D array")
    img = img / 255.0
    ys, xs = np.where(img > 0.1)
    if len(ys) == 0:
        raise ValueError("canvas is empty")
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(img.shape)
    padded = np.zeros((side, side))
    oy, ox = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
    padded[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    margin = side // 32
    padded = np.pad(padded, margin)
    idx = np.minimum((np.arange(64) * padded.shape[0] / 64).astype(int), padded.shape[0] - 1)
    big = padded[np.ix_(idx, idx)]
    small = big.reshape(8, 8, 8, 8).mean(axis=(1, 3))
    if small.max() > 0:
        small = small / small.max()
    return small.reshape(1, 64)

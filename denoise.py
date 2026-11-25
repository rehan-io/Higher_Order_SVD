"""
denoise.py - Gaussian noise removal with patch-based HOSVD (from hosvd.py).

For every reference patch in the image:
  1. Find the K most similar patches nearby.
  2. Stack them into a 4D tensor (patch x patch x colour x K).
  3. Compute its HOSVD.
  4. Set small core values to zero (they are mostly noise).
  5. Rebuild the patches and average them back into the image.

Usage
  python denoise.py NOISY_DIR CLEAN_DIR      run on your dataset with PSNR (results/ folder)
  python denoise.py NOISY_DIR                only denoise, no PSNR (results/ folder)
"""
import os
import sys
import time

import numpy as np
from PIL import Image

from hosvd import hosvd, reconstruct


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def load_image(path):
    """Read an image as float in [0, 1] with shape (H, W, C)."""
    img = np.asarray(Image.open(path)).astype(np.float64) / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    return img[:, :, :3]                     # drop alpha channel if present


def save_image(img, path):
    img = np.clip(img, 0, 1)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray((img * 255).round().astype(np.uint8)).save(path)


def psnr(clean, test):
    mse = np.mean((clean - test) ** 2)
    return 10 * np.log10(1.0 / mse)


def estimate_noise_sigma(img):
    """Estimate the Gaussian noise standard deviation from the image itself.

    Differences between diagonal neighbours are almost pure noise in a natural
    image, so  sigma ~ median(|HH|) / 0.6745.
    """
    g = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2, :]   # even size
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    return np.median(np.abs(hh)) / 0.6745


def threshold_core(S, sigma):
    """Set core values smaller than the noise threshold to zero."""
    tau = sigma * np.sqrt(2 * np.log(S.size))
    S = S.copy()
    S[np.abs(S) < tau] = 0
    return S


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def denoise_hosvd(noisy, sigma, patch=8, stride=4, search=15, K=30):
    """Denoise an (H, W, C) image in [0, 1] with patch-group HOSVD."""
    H, W, C = noisy.shape
    p = patch
    out = np.zeros_like(noisy)
    weight = np.zeros_like(noisy)

    # Reference patch positions (make sure the last row/column is covered)
    ys = list(range(0, H - p + 1, stride))
    xs = list(range(0, W - p + 1, stride))
    if ys[-1] != H - p:
        ys.append(H - p)
    if xs[-1] != W - p:
        xs.append(W - p)

    for i in ys:
        for j in xs:
            ref = noisy[i:i + p, j:j + p, :]

            # 1. Search window around the reference patch
            y0, y1 = max(0, i - search), min(H - p, i + search)
            x0, x1 = max(0, j - search), min(W - p, j + search)
            window = noisy[y0:y1 + p, x0:x1 + p, :]
            # all p x p patches in the window: shape (ny, nx, C, p, p)
            cands = np.lib.stride_tricks.sliding_window_view(window, (p, p), axis=(0, 1))
            dist = np.sum((cands - ref.transpose(2, 0, 1)) ** 2, axis=(2, 3, 4))

            # 2. K most similar patches (the reference itself has distance 0)
            k = min(K, dist.size)
            best = np.argsort(dist, axis=None)[:k]
            pos_y, pos_x = np.unravel_index(best, dist.shape)
            pos_y, pos_x = pos_y + y0, pos_x + x0

            # 3. Stack into a 4D tensor (p, p, C, k) and take its HOSVD
            group = np.stack([noisy[y:y + p, x:x + p, :] for y, x in zip(pos_y, pos_x)],
                             axis=3)
            S, U = hosvd(group)

            # 4. Remove noise in the core, then rebuild the patches
            S = threshold_core(S, sigma)
            clean_group = reconstruct(S, U)

            # 5. Put the patches back. Groups with fewer surviving values
            #    are more reliable, so they get a larger weight.
            w = 1.0 / (1.0 + np.count_nonzero(S))
            for idx, (y, x) in enumerate(zip(pos_y, pos_x)):
                out[y:y + p, x:x + p, :] += w * clean_group[:, :, :, idx]
                weight[y:y + p, x:x + p, :] += w

    return np.clip(out / weight, 0, 1)


def denoise_image(noisy):
    """Estimate the noise level, then denoise."""
    sigma = estimate_noise_sigma(noisy)
    return denoise_hosvd(noisy, sigma), sigma


# ---------------------------------------------------------------------------
# Running on a dataset folder
# ---------------------------------------------------------------------------

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def run_noisy_only(noisy_dir, out_dir):
    """Denoise every image in a folder (no clean images, so no PSNR)."""
    os.makedirs(out_dir, exist_ok=True)
    for fname in list_images(noisy_dir):
        t = time.time()
        den, sigma = denoise_image(load_image(os.path.join(noisy_dir, fname)))
        save_image(den, os.path.join(out_dir, os.path.splitext(fname)[0] + "_denoised.png"))
        print(f"{fname}: sigma={sigma * 255:.1f}/255 | {time.time() - t:.0f}s")


def run_folder(noisy_dir, clean_dir, out_dir):
    noisy_files = list_images(noisy_dir)
    clean_files = list_images(clean_dir)
    if len(clean_files) < len(noisy_files):
        sys.exit(f"Error: '{clean_dir}' has {len(clean_files)} images but '{noisy_dir}' "
                 f"has {len(noisy_files)}. The second argument must be the folder of "
                 f"clean (original) images.")
    os.makedirs(out_dir, exist_ok=True)
    scores = []

    for n, fname in enumerate(noisy_files):
        # same file name if possible, otherwise the n-th file in sorted order
        cname = fname if fname in clean_files else clean_files[n]
        noisy = load_image(os.path.join(noisy_dir, fname))
        clean = load_image(os.path.join(clean_dir, cname))
        if clean.shape[2] != noisy.shape[2]:          # grey vs colour mismatch
            clean = clean.mean(axis=2, keepdims=True)
            noisy = noisy.mean(axis=2, keepdims=True)

        t = time.time()
        den, sigma = denoise_image(noisy)
        save_image(den, os.path.join(out_dir, os.path.splitext(fname)[0] + "_denoised.png"))

        before, after = psnr(clean, noisy), psnr(clean, den)
        scores.append((before, after))
        print(f"{fname} (clean: {cname}): sigma={sigma * 255:.1f}/255 | "
              f"PSNR {before:.2f} -> {after:.2f} dB | {time.time() - t:.0f}s")

    before, after = np.mean(scores, axis=0)
    print(f"\nAverage over {len(scores)} images: PSNR {before:.2f} -> {after:.2f} dB")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_folder(noisy_dir=sys.argv[1], clean_dir=sys.argv[2], out_dir="results")
    elif len(sys.argv) == 2:
        run_noisy_only(noisy_dir=sys.argv[1], out_dir="results")
    else:
        print("Usage:\n  python denoise.py NOISY_DIR CLEAN_DIR\n  python denoise.py NOISY_DIR")

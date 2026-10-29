#!/usr/bin/env python3
"""
Patch-based HOSVD denoising (grayscale + RGB).
Usage:
    python hosvd_denoise.py noisy.png denoised.png \
        [--sigma S] [--patch_size P] [--stride T] \
        [--max_patches K] [--search_radius R]
"""

import argparse
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------------
# 1. Tensor primitives (unfold / fold / mode-n product)
# ----------------------------------------------------------------------
def unfold(tensor, mode):
    moved = np.moveaxis(tensor, mode, 0)
    return moved.reshape(moved.shape[0], -1)


def fold(matrix, mode, shape):
    moved_shape = (shape[mode],) + tuple(shape[:mode] + shape[mode + 1:])
    moved = matrix.reshape(moved_shape)
    return np.moveaxis(moved, 0, mode)


def mode_n_product(tensor, matrix, mode):
    unfolded = unfold(tensor, mode)
    multiplied = matrix @ unfolded
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(multiplied, mode, tuple(new_shape))


# ----------------------------------------------------------------------
# 2. HOSVD (hard-coded, only numpy.linalg.svd)
# ----------------------------------------------------------------------
def hosvd(tensor, ranks=None):
    ndim = tensor.ndim
    if ranks is None:
        ranks = [None] * ndim
    factors = []
    for n in range(ndim):
        mat = unfold(tensor, n)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        if ranks[n] is not None:
            U = U[:, :ranks[n]]
        factors.append(U)
    core = tensor.copy()
    for n in range(ndim):
        core = mode_n_product(core, factors[n].T, n)
    return core, factors


def hosvd_reconstruct(core, factors):
    rec = core.copy()
    for n, U in enumerate(factors):
        rec = mode_n_product(rec, U, n)
    return rec


# ----------------------------------------------------------------------
# 3. Noise-level estimation (MAD of high-pass residual)
# ----------------------------------------------------------------------
def estimate_sigma(img):
    blurred = gaussian_filter(img, sigma=1.0)
    high = img - blurred
    mad = np.median(np.abs(high - np.median(high)))
    return 1.4826 * mad


# ----------------------------------------------------------------------
# 4. Main denoiser -- handles both 2-D and 3-D (RGB) input
# ----------------------------------------------------------------------
def denoise_image(noisy, patch_size=8, stride=None, search_radius=20,
                  max_patches=32, sigma=None, verbose=True):
    # ---------- RGB branch: denoise each channel independently ----------
    if noisy.ndim == 3:
        H, W, C = noisy.shape
        out = np.empty_like(noisy, dtype=np.float64)
        ch_iter = range(C)
        if verbose:
            ch_iter = tqdm(ch_iter, desc="channels", unit="ch")
        for c in ch_iter:
            out[..., c] = denoise_image(
                noisy[..., c],
                patch_size=patch_size,
                stride=stride,
                search_radius=search_radius,
                max_patches=max_patches,
                sigma=sigma,
                verbose=verbose,
            )
        return out

    # ---------- 2-D path -------------------------------------------------
    if stride is None:
        stride = patch_size // 2
    if sigma is None:
        sigma = estimate_sigma(noisy)

    H, W = noisy.shape

    # All patches via a zero-copy sliding-window view, subsampled by `stride`
    patches_grid = np.lib.stride_tricks.sliding_window_view(
        noisy, (patch_size, patch_size))[::stride, ::stride]
    nR, nC = patches_grid.shape[:2]
    N = nR * nC
    if N == 0:
        raise ValueError("Image smaller than patch size.")

    # Materialise as (N, p*p) for fast vectorised distances
    flat = np.ascontiguousarray(patches_grid).reshape(N, -1).astype(np.float64)
    norms = np.sum(flat ** 2, axis=1)

    p = patch_size
    K = min(max_patches, N)
    threshold_svd = sigma * np.sqrt(2.0 * np.log(p * p * K))
    w = max(1, search_radius // stride)     # search radius in grid units

    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)

    pbar = tqdm(total=N, disable=not verbose,
                desc=f"  denoise p={p} stride={stride} w={w} K={K}",
                unit="patch", leave=False)
    t0 = time.time()

    for gi in range(nR):
        i_lo = max(0, gi - w)
        i_hi = min(nR, gi + w + 1)
        for gj in range(nC):
            idx = gi * nC + gj
            j_lo = max(0, gj - w)
            j_hi = min(nC, gj + w + 1)

            # Candidate patch indices in the local neighbourhood
            ii, jj = np.meshgrid(np.arange(i_lo, i_hi),
                                 np.arange(j_lo, j_hi),
                                 indexing="ij")
            nbr_idx = (ii * nC + jj).ravel()

            # Squared distances ||a||^2 + ||b||^2 - 2 a·b
            ref = flat[idx]
            d = norms[nbr_idx] + norms[idx] - 2.0 * (flat[nbr_idx] @ ref)
            d = np.maximum(d, 0.0)

            # k nearest (reference always included)
            k = min(max_patches, len(nbr_idx))
            sel = nbr_idx[np.argsort(d)[:k]]
            if idx not in sel:
                sel = np.append(sel[:k - 1], idx)

            # Build (p, p, K') tensor and denoise it
            tensor = flat[sel].T.reshape(p, p, -1)
            core, factors = hosvd(tensor)
            core[np.abs(core) < threshold_svd] = 0.0
            rec = hosvd_reconstruct(core, factors)

            # Reference slice (its position in `sel` is guaranteed)
            pos = int(np.where(sel == idx)[0][0])
            patch_out = rec[:, :, pos]

            r0 = gi * stride
            c0 = gj * stride
            acc[r0:r0 + p, c0:c0 + p] += patch_out
            cnt[r0:r0 + p, c0:c0 + p] += 1.0
            pbar.update(1)

    pbar.close()
    if verbose:
        print(f"  channel finished in {time.time() - t0:.1f}s")

    cnt[cnt == 0] = 1.0
    return acc / cnt


# ----------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Patch-based HOSVD denoiser")
    parser.add_argument("input", help="Noisy PNG (grayscale or RGB)")
    parser.add_argument("output", help="Denoised PNG")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Noise std (per-channel estimate if omitted)")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Square patch side (default: 8)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between patches (default: patch_size//2)")
    parser.add_argument("--max_patches", type=int, default=32,
                        help="Max similar patches in a tensor (default: 32)")
    parser.add_argument("--search_radius", type=int, default=20,
                        help="Spatial search radius in pixels (default: 20)")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable progress bars")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    noisy = np.asarray(img, dtype=np.float64)
    print(f"input: {noisy.shape[0]}×{noisy.shape[1]}, mode=RGB")

    denoised = denoise_image(
        noisy,
        patch_size=args.patch_size,
        stride=args.stride,
        search_radius=args.search_radius,
        max_patches=args.max_patches,
        sigma=args.sigma,
        verbose=not args.quiet,
    )

    Image.fromarray(np.clip(denoised, 0, 255).astype(np.uint8)).save(args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
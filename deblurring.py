
import numpy as np

from numerical_methods import (
    build_blur_matrix,
    direct_restore,
    regularized_least_squares,
    estimate_condition_number
)


def mse(original, restored):
    return float(np.mean((original - restored) ** 2))


def psnr(original, restored):
    error = mse(original, restored)

    if error < 1e-15:
        return float("inf")

    return float(10 * np.log10(1 / error))


def restore_image(image, method, regularization=0.025):

    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square and grayscale.")

    n = image.shape[0]
    B = build_blur_matrix(n)

    if method == "Gauss Elimination":
        restored = direct_restore(B, image, "gauss")

    elif method == "Gauss-Jordan":
        restored = direct_restore(B, image, "jordan")

    elif method == "Regularized Least Squares":
        restored = regularized_least_squares(
            B, image, regularization
        )

    else:
        raise ValueError("Unknown restoration method.")

    A = B.T @ B + regularization * np.eye(n)

    largest, smallest, condition = estimate_condition_number(A)

    return {
        "image": np.clip(restored, 0, 1),
        "largest": largest,
        "smallest": smallest,
        "condition": condition
    }


def compare_quality(original, degraded, restored):

    return {
        "mse_before": mse(original, degraded),
        "mse_after": mse(original, restored),
        "psnr_before": psnr(original, degraded),
        "psnr_after": psnr(original, restored)
    }

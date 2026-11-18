
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from image_processing import (
    load_image,
    make_demo_image,
    blur_image,
    add_noise,
    save_image
)

from deblurring import restore_image, compare_quality


def main():

    parser = argparse.ArgumentParser(
        description="Photo-Signal Deblur Lab"
    )

    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--noise", type=float, default=0.02)

    parser.add_argument(
        "--method",
        choices=[
            "Gauss Elimination",
            "Gauss-Jordan",
            "Regularized Least Squares"
        ],
        default="Regularized Least Squares"
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.025
    )

    args = parser.parse_args()

    if not 8 <= args.size <= 64:
        parser.error("Image size must be between 8 and 64.")

    if not 0 <= args.noise <= 0.2:
        parser.error("Noise must be between 0 and 0.2.")

    if args.regularization <= 0:
        parser.error("Regularization must be greater than zero.")

    print("\n================================")
    print("       PHOTO-SIGNAL DEBLUR LAB")
    print("================================\n")

    if args.image:
        original = load_image(args.image, args.size)
        print("Custom image loaded successfully.")
    else:
        original = make_demo_image(args.size)
        print("Demo image generated successfully.")

    print("Applying blur...")
    blurred = blur_image(original)

    print("Adding noise...")
    degraded = add_noise(blurred, args.noise)

    print("Restoring image using:", args.method)

    result = restore_image(
        degraded,
        args.method,
        args.regularization
    )

    restored = result["image"]

    metrics = compare_quality(
        original,
        degraded,
        restored
    )

    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    save_image(original, output_folder / "original.png")
    save_image(degraded, output_folder / "blurred_noisy.png")
    save_image(restored, output_folder / "restored.png")

    print("\n========== RESULTS ==========")

    print(f"MSE Before Restoration: {metrics['mse_before']:.6f}")
    print(f"MSE After Restoration:  {metrics['mse_after']:.6f}")

    print(f"PSNR Before Restoration: {metrics['psnr_before']:.2f} dB")
    print(f"PSNR After Restoration:  {metrics['psnr_after']:.2f} dB")

    print(f"\nLargest Eigenvalue: {result['largest']:.6f}")
    print(f"Smallest Eigenvalue: {result['smallest']:.6f}")
    print(f"Condition Number:    {result['condition']:.6f}")

    print("\nImages saved in the outputs folder.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    images = [original, degraded, restored]
    titles = ["Original Image", "Blurred + Noisy", "Restored Image"]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

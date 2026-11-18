
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


def load_image(source, size=32):
    if isinstance(source, Image.Image):
        img = source.convert("L")
    else:
        img = Image.open(source).convert("L")

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=float) / 255.0


def make_demo_image(size=32):
    img = Image.new("L", (size, size), 25)
    draw = ImageDraw.Draw(img)

    margin = max(2, size // 8)

    draw.rectangle(
        (margin, margin, size - margin - 1, size - margin - 1),
        fill=175
    )

    draw.ellipse(
        (size // 4, size // 4, 3 * size // 4, 3 * size // 4),
        fill=245
    )

    draw.line(
        (2, size // 2, size - 3, size // 2),
        fill=80,
        width=max(1, size // 16)
    )

    return np.asarray(img, dtype=float) / 255.0


def blur_image(image):
    kernel = np.ones(3) / 3

    padded = np.pad(image, ((0, 0), (1, 1)), mode="edge")

    horizontal = sum(
        kernel[k] * padded[:, k:k + image.shape[1]]
        for k in range(3)
    )

    padded = np.pad(horizontal, ((1, 1), (0, 0)), mode="edge")

    vertical = sum(
        kernel[k] * padded[k:k + image.shape[0], :]
        for k in range(3)
    )

    return np.clip(vertical, 0, 1)


def add_noise(image, level=0.02, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, level, image.shape)

    return np.clip(image + noise, 0, 1)


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pixels = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    Image.fromarray(pixels).save(path)

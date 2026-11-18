
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from image_processing import (
    load_image, make_demo_image, blur_image, add_noise
)
from deblurring import restore_image, compare_quality


st.set_page_config(
    page_title="Photo-Signal Deblur Lab",
    page_icon="📷",
    layout="wide"
)

st.title("📷 Photo-Signal Deblur Lab")
st.subheader("Numerical Image Restoration and Performance Analysis")

st.write(
    "Compare numerical algorithms for image restoration "
    "using mathematical methods."
)

st.sidebar.header("Experiment Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image", type=["png", "jpg", "jpeg"]
)

size = st.sidebar.slider("Image Resolution", 8, 64, 32, 8)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.02, 0.01)

method = st.sidebar.selectbox(
    "Restoration Method",
    ["Regularized Least Squares", "Gauss Elimination", "Gauss-Jordan"]
)

lam = st.sidebar.slider(
    "Regularization Parameter", 0.001, 0.2, 0.025, 0.001
)

if uploaded_file:
    original = load_image(uploaded_file, size)
else:
    original = make_demo_image(size)

blurred = blur_image(original)
degraded = add_noise(blurred, noise_level)

st.header("Image Processing")

c1, c2, c3 = st.columns(3)

with c1:
    st.image(original, caption="Original Image", clamp=True)

with c2:
    st.image(degraded, caption="Blurred + Noisy Image", clamp=True)

if st.button("Restore Image", type="primary"):
    try:
        result = restore_image(degraded, method, lam)
        restored = result["image"]
        metrics = compare_quality(original, degraded, restored)

        st.session_state["restored"] = restored
        st.session_state["metrics"] = metrics
        st.session_state["result"] = result

    except Exception as error:
        st.error(f"Restoration error: {error}")

with c3:
    if "restored" in st.session_state:
        st.image(
            st.session_state["restored"],
            caption="Restored Image",
            clamp=True
        )
    else:
        st.info("Click Restore Image.")

if "metrics" in st.session_state:
    st.divider()
    st.header("Restoration Performance")

    metrics = st.session_state["metrics"]
    result = st.session_state["result"]

    a, b = st.columns(2)

    a.metric("MSE Before", f"{metrics['mse_before']:.6f}")
    a.metric("MSE After", f"{metrics['mse_after']:.6f}")

    b.metric("PSNR Before", f"{metrics['psnr_before']:.2f} dB")
    b.metric("PSNR After", f"{metrics['psnr_after']:.2f} dB")

    st.subheader("Eigenvalue Analysis")

    x, y, z = st.columns(3)
    x.metric("Largest Eigenvalue", f"{result['largest']:.6f}")
    y.metric("Smallest Eigenvalue", f"{result['smallest']:.6f}")
    z.metric("Condition Number", f"{result['condition']:.4f}")

    output = BytesIO()
    pixels = (np.clip(st.session_state["restored"], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output, format="PNG")

    st.download_button(
        "Download Restored Image",
        output.getvalue(),
        "restored_image.png",
        "image/png"
    )

st.divider()

# AUTOMATIC ALGORITHM COMPARISON

st.header("🔬 Automatic Algorithm Comparison")

st.write(
    "Test all three numerical methods on the same degraded image "
    "and compare their restoration performance."
)

if st.button("Run Complete Algorithm Comparison"):

    methods = [
        "Gauss Elimination",
        "Gauss-Jordan",
        "Regularized Least Squares"
    ]

    records = []

    progress = st.progress(0)
    status = st.empty()

    for index, current_method in enumerate(methods):

        status.write(f"Testing: {current_method}")

        try:
            output = restore_image(degraded, current_method, lam)
            restored = output["image"]

            score = compare_quality(original, degraded, restored)

            records.append({
                "Method": current_method,
                "MSE Before": score["mse_before"],
                "MSE After": score["mse_after"],
                "PSNR Before": score["psnr_before"],
                "PSNR After": score["psnr_after"],
                "Condition Number": output["condition"]
            })

        except Exception as error:
            st.warning(f"{current_method} failed: {error}")

        progress.progress((index + 1) / len(methods))

    if records:
        st.session_state["comparison"] = records

    status.success("Comparison completed!")

if "comparison" in st.session_state:

    records = st.session_state["comparison"]
    df = pd.DataFrame(records)

    st.subheader("Algorithm Performance Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("MSE Comparison")

    mse_data = df.set_index("Method")[["MSE Before", "MSE After"]]
    st.bar_chart(mse_data)

    st.subheader("PSNR Comparison")

    psnr_data = df.set_index("Method")[["PSNR Before", "PSNR After"]]
    st.bar_chart(psnr_data)

    best = df.loc[df["MSE After"].idxmin(), "Method"]

    st.success(f"Best method for this experiment (lowest MSE): {best}")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Experiment Report (CSV)",
        csv,
        "algorithm_comparison.csv",
        "text/csv"
    )

st.divider()

st.header("About the Project")

st.write("""
This project demonstrates numerical image restoration using
Gauss Elimination, Gauss-Jordan and Regularized Least Squares.

Power Method and Inverse Power Method are used for eigenvalue
and condition number analysis.

MSE and PSNR are used to evaluate restoration quality.
""")

st.caption("Photo-Signal Deblur Lab | Numerical Analysis Programming")

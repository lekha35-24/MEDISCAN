import streamlit as st
import cv2
import numpy as np
from PIL import Image

from counterfeit_model import predict_counterfeit, load_model, grad_cam, transform, device
from batch_ocr import extract_batch_number, get_text_density
from expiry_checker import check_expiry_validity


model = load_model()

HIGH_CONF = 0.90
MED_CONF = 0.80
TEXT_DENSITY_THRESHOLD = 25


def decide(result, conf, batch, expiry_status, is_back_side):
    if is_back_side:
        if batch is None and expiry_status == "NOT FOUND":
            trust_note = " (OCR unreadable — relying on visual model)"
        else:
            trust_note = ""

        if expiry_status == "EXPIRED":
            return "REJECTED: Expired"

        if result == "Fake" and conf > HIGH_CONF:
            return "HIGHLY SUSPICIOUS" + trust_note
        if result == "Fake" and conf > MED_CONF:
            return "SUSPICIOUS" + trust_note
        if result == "Real" and conf > HIGH_CONF:
            return "LIKELY GENUINE" + trust_note
        return "UNCERTAIN" + trust_note
    else:
        if result == "Fake" and conf > HIGH_CONF:
            return "SUSPICIOUS (visual only)"
        if result == "Real" and conf > HIGH_CONF:
            return "LIKELY GENUINE (visual only)"
        return "UNCERTAIN (visual only)"


# ✅ FIXED Grad-CAM for Streamlit
def run_grad_cam_streamlit(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        heatmap = grad_cam(model, img_tensor)

        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )

        img_resized = cv2.resize(img, (224, 224))
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="Grad-CAM",
                 use_container_width=True)

    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")


# ---------------- UI ---------------- #
st.title("Medicine Checker")

path = st.text_input("Enter image path (same as your main.py):")

if path:
    img = cv2.imread(path)

    if img is None:
        st.error("Image not found")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image")

        text_density = get_text_density(path)
        is_back_side = text_density >= TEXT_DENSITY_THRESHOLD

        st.write("Text density:", text_density)
        st.write("Side:", "Back" if is_back_side else "Front")

        batch = extract_batch_number(path)

        # ✅ MAIN FIX HERE
        expiry_status, expiry_conf, expiry_date = check_expiry_validity(path)

        st.write("Batch:", batch if batch else "Not detected")
        st.write("Expiry Status:", expiry_status)
        st.write("Expiry Date:", expiry_date if expiry_date else "Not detected")

        result, conf = predict_counterfeit(model, img)

        st.write("Prediction:", result)
        st.write("Confidence:", round(conf, 2))

        final = decide(result, conf, batch, expiry_status, is_back_side)

        st.subheader("Final Result")
        st.write(final)

        run_grad_cam_streamlit(img)
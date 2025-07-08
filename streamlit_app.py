import os
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="YOLO Boat Detection", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# YOLO MODEL LOADING
# ============================

@st.cache_resource
def load_yolo_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# ============================
# IMAGE PROCESSING
# ============================

def predict_yolo(model, image, conf_threshold=0.5):
    try:
        results = model(image, conf=conf_threshold)
        return results
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return None

def draw_yolo_detections(image, results):
    image_draw = image.copy()
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Boat: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_draw, (x1, y1 - th - 5), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(image_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image_draw

# ============================
# MAIN APP
# ============================

def main():
    st.markdown('<div class="main-header">🚢 YOLOv8 Boat Detection</div>', unsafe_allow_html=True)

    # Sidebar - Model loading
    st.sidebar.header("YOLOv8 Model")
    yolo_model_file = st.sidebar.file_uploader("Upload YOLOv8 Model (.pt)", type=["pt"])

    yolo_model = None
    if yolo_model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(yolo_model_file.read())
            yolo_model_path = tmp.name
            yolo_model = load_yolo_model(yolo_model_path)

    if yolo_model is not None:
        st.sidebar.success("✅ YOLOv8 model loaded successfully.")
    else:
        st.sidebar.warning("Upload a YOLOv8 `.pt` file to begin.")

    st.sidebar.subheader("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

    uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images and yolo_model:
        if st.button("🔍 Run Detection"):
            for idx, img_file in enumerate(uploaded_images):
                image = Image.open(img_file).convert("RGB")
                image_np = np.array(image)

                st.markdown(f"### 📷 Image {idx + 1}: {img_file.name}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image_np, caption="Original Image", use_column_width=True)

                with col2:
                    results = predict_yolo(yolo_model, image_np, conf_threshold)
                    result_img = draw_yolo_detections(image_np, results)
                    st.image(result_img, caption="Detected Boats", use_column_width=True)

                    if results and results[0].boxes is not None:
                        boat_count = len(results[0].boxes)
                        avg_conf = float(results[0].boxes.conf.mean())
                        st.metric("Boats Detected", boat_count)
                        st.metric("Average Confidence", f"{avg_conf:.2f}")
                    else:
                        st.metric("Boats Detected", 0)
                        st.metric("Average Confidence", "0.00")

if __name__ == "__main__":
    main()

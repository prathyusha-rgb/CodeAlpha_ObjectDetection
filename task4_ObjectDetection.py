import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Object Detection", page_icon="📷", layout="wide")

st.title("📷 Object Detection & Tracking")
st.markdown("**CodeAlpha AI Internship — Task 4**")
st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Downloads automatically on first run

model = load_model()

option = st.radio("Choose Input Type:", ["📁 Upload Image", "🎥 Upload Video", "📷 Webcam (Live)"])

if option == "📁 Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        img_array = np.array(image)

        with st.spinner("Detecting objects..."):
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            results = model(img_array)
            annotated = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated, caption="Detected Objects", use_column_width=True)

        # Show detected labels
        names = results[0].names
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            labels = [names[int(cls)] for cls in boxes.cls]
            st.success(f"✅ Detected: {', '.join(set(labels))}")
        else:
            st.info("No objects detected.")

elif option == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stop = st.button("⛔ Stop")

        st.info("Processing video frame by frame...")
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_column_width=True)

        cap.release()
        os.unlink(tfile.name)
        st.success("✅ Video processing complete!")

elif option == "📷 Webcam (Live)":
    st.warning("⚠️ Webcam works best when running locally with: streamlit run app.py")
    run = st.button("▶️ Start Webcam")
    stop = st.button("⛔ Stop Webcam")
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access webcam.")
                break
            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_column_width=True)
        cap.release()

st.markdown("---")
st.caption("Powered by YOLOv8 (Ultralytics) | CodeAlpha Internship")

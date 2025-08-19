import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO("best.pt")

st.title("Pakistani Number Plate Detection")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(np.array(image))
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Plates", use_column_width=True)

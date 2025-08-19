import gradio as gr
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("best.pt")  # update path if needed

def detect_number_plate(image):
    results = model.predict(image)
    annotated = results[0].plot()  # draw boxes
    return annotated

app = gr.Interface(
    fn=detect_number_plate,
    inputs=gr.Image(type="numpy", label="Upload Car Image"),
    outputs=gr.Image(type="numpy", label="Detected Plates"),
    title="Pakistani Number Plate Detection 🚗",
    description="Upload a car image, and the YOLO model will detect Pakistani number plates."
)

app.launch()

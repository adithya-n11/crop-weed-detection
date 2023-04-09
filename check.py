import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv5 model
model = torch.load('best.pt')

# Define the Streamlit app
def app():
    st.title("YOLOv5 Object Detection")

    # Allow the user to upload an image
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        # Load the image
        image = Image.open(image_file)

        # Make a prediction on the image
        predictions = model(image)

        # Draw the bounding boxes on the image
        boxes = predictions.xyxy[0].cpu().numpy()
        scores = predictions.xyxy[0, :, 4].cpu().numpy()
        labels = predictions.xyxy[0, :, 5].cpu().numpy()
        for i in range(len(boxes)):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = boxes[i]
                label = str(labels[i])
                image = cv2.rectangle(np.array(image), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                image = cv2.putText(np.array(image), label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with bounding boxes
        st.image(image, caption="Object Detection", use_column_width=True)

# Run the app
app()

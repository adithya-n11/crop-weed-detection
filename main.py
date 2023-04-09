#Initial Setup
import torch
import os
from IPython.display import Image, clear_output  # to display images
from yolov5.detect import run
from roboflow import Roboflow
# print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")



# Loading Dataset
# rf = Roboflow(api_key="8cmyFXT2OJJSOqYXpyeo")
# project = rf.workspace("project-kuuew").project("lettuce-weed-4")
# dataset = project.version(1).download("yolov5")

# Upload Test data
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import datetime

# Create a Tkinter window
root = tk.Tk()
root.withdraw()

# Open a file dialog box and get the path of the selected file
file_path = filedialog.askopenfilename()

# Load the image file
image = Image.open(file_path)

# Create a directory for uploaded images if it doesn't exist
upload_dir = "Upload"
if not os.path.exists(upload_dir):
    os.mkdir(upload_dir)

# Generate a unique file name based on the current date and time
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = current_time + "_" + file_path.split("/")[-1]

# Save the image to the upload directory with the unique file name
image.save(os.path.join(upload_dir, file_name))

# Get the path of the saved image file
saved_file_path = os.path.join(upload_dir, file_name)

# print("Path of uploaded image:", file_path)
# print("Path of saved image:", saved_file_path)



# Testing

run(source=saved_file_path,project='Results')

#firebase

# without firebase + yolo

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image, ImageDraw
#
# # Load the YOLOv5 model
# model = tf.saved_model.load('/best.pt')
#
# @st.cache(allow_output_mutation=True)
# def predict(image):
#     # Preprocess the image
#     tensor = tf.keras.preprocessing.image.img_to_array(image)
#     resized = tf.image.resize(tensor, [416, 416])
#     normalized = resized / 255.0
#     expanded = np.expand_dims(normalized, axis=0)
#
#     # Make a prediction
#     predictions = model(expanded)
#
#     # Convert the predictions to a list of objects
#     results = []
#     for output in predictions:
#         boxes = output[:, :4].numpy()
#         scores = output[:, 4].numpy()
#         classes = output[:, 5].numpy().astype(int)
#         for i in range(len(scores)):
#             if scores[i] >= 0.5:
#                 ymin, xmin, ymax, xmax = boxes[i]
#                 score = scores[i]
#                 class_id = classes[i]
#                 results.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'score': score, 'class_id': class_id})
#
#     return results
#
# def main():
#     st.title('YOLOv5 Demo')
#
#     # Upload an image
#     uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
#
#     if uploaded_file is not None:
#         # Load and display the image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Make predictions and display the results
#         predictions = predict(image)
#         draw = ImageDraw.Draw(image)
#         for prediction in predictions:
#             xmin = int(prediction['xmin'] * image.width)
#             ymin = int(prediction['ymin'] * image.height)
#             xmax = int(prediction['xmax'] * image.width)
#             ymax = int(prediction['ymax'] * image.height)
#             score = prediction['score']
#             class_id = prediction['class_id']
#             label = f'Class {class_id}, Score {score:.2f}'
#             draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
#             draw.text([xmin, ymin-16], label, fill='white')
#         st.image(image, caption='YOLOv5 Predictions', use_column_width=True)
#
# if __name__ == '__main__':
#     main()


# import subprocess
#
# command = 'python yolov5/train.py --img 640 --batch 16 --epochs 3 --data lettuce-weed-4-1/data.yaml --weights best.pt --cache'
# subprocess.call(command.split())
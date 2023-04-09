import streamlit as st
import asyncio
import os
from PIL import Image
from yolov5.detect import run

# Define an async function to perform inference on a single image
async def do_inference(image_path):
    # Replace this with your actual inference code
    # This is just a dummy example that returns the original image
    result_image,result_path = run(source=image_path) # Replace this with your actual result image
    return result_image, result_path


# Define an async function to perform inference on all uploaded images
async def process_uploaded_images(uploaded_images):
    result_images = []
    for uploaded_image in uploaded_images:
        # Store the uploaded file on the local system
        with open(os.path.join("Upload", uploaded_image.name), "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Open the image using PIL
        image = Image.open(uploaded_image)

        # Define a two-column layout to display the images
        col1, col2 = st.columns(2)

        # Display the uploaded image on the left
        with col1:
            st.image(image, caption=f"Uploaded Image ({uploaded_image.name})", use_column_width=True)

        # Call the async function to perform inference
        task = asyncio.create_task(do_inference(os.path.join("Upload", uploaded_image.name)))

        # Display a spinner while inference is being performed
        with col2:
            with st.spinner(f'Performing inference on {uploaded_image.name}...'):
                # Wait for the inference to complete and get the result
                result_image, result_path = await task

            # Display the result image on the right

            st.image(result_image, caption=f"Result Image ({uploaded_image.name})", use_column_width=True)

        # Add the result image path to the list of result images
        result_images.append(result_path)

    return result_images

# Define a function to run the Streamlit app
async def run_app():
    # Create a sidebar
    st.sidebar.title("Upload Images")

    # Allow user to upload multiple images
    uploaded_images = st.sidebar.file_uploader(label="Choose one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images is not None and len(uploaded_images) > 0:
        # Call the async function to perform inference on all uploaded images
        task = asyncio.create_task(process_uploaded_images(uploaded_images))

        # Display a spinner while inference is being performed
        with st.spinner('Performing inference on uploaded images...'):
            # Wait for the inference to complete and get the result
            result_images = await task

        # Display a message indicating the number of images processed
        st.write(f"Processed {len(result_images)} images.")
    else:
        st.write("No images uploaded yet.")

# Run the Streamlit app
if __name__ == '__main__':
    asyncio.run(run_app())

import os
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2

# Load the trained model (replace with the correct path to your model)
model = tf.keras.models.load_model('symbol_model.h5')

# Class names (symbols)
class_names = ['+', '-', '*', '/']

# Streamlit UI setup
st.set_page_config(page_title="Handwritten Symbol Recognition", layout="wide")

st.sidebar.title("🖌️ How to Use")
st.sidebar.write("1. Draw a mathematical symbol in the white box.")
st.sidebar.write("2. Click outside the box to process.")
st.sidebar.write("3. The predicted symbol will appear below.")
st.sidebar.write("4. Click 'Clear' to redraw.")

st.title("✍️ Handwritten Symbol Recognition")

# Layout for UI components
col1, col2 = st.columns([2, 2])

with col1:
    st.subheader("Draw a Mathematical Symbol")
    
    # Create a drawing canvas
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Button to clear canvas
    if st.button("Clear Drawing"):
        st.session_state["canvas_key"] = str(np.random.randint(1000))  # Update the key
        st.rerun()  # Forces a rerun to clear the canvas

with col2:
    # Only proceed if the canvas has data
    if canvas_result.image_data is not None and np.any(canvas_result.image_data):
        # Convert image to grayscale
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)

        # Invert colors to match training data (if needed)
        img = cv2.bitwise_not(img)

        # Apply threshold to remove noise (optional, depending on your dataset)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # Resize to 28x28 pixels (the model expects 28x28 images)
        img = cv2.resize(img, (28, 28))

        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0

        # Add necessary dimensions to match model input (1, 28, 28, 1)
        img = np.expand_dims(img, axis=-1)  # (28, 28, 1)
        img = np.expand_dims(img, axis=0)   # (1, 28, 28, 1)

        # Make prediction using the model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Display the predicted symbol
        st.subheader(f"🔢 Predicted Symbol: **{class_names[predicted_class]}**")

        # Display confidence scores
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0], color="blue")
        ax.set_xlabel("Symbols")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)
    else:
        st.subheader("✍️ Please draw a symbol before predicting.")
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Load the trained model (no retraining needed)
model = tf.keras.models.load_model('digit_model.keras')

# Streamlit UI
st.set_page_config(page_title="Digit Recognition", layout="wide")

# Sidebar with instructions
st.sidebar.title("🖌️ How to Use")
st.sidebar.write("1. Draw a digit in the white box.")
st.sidebar.write("2. Click outside the box to process.")
st.sidebar.write("3. The predicted digit will appear below.")
st.sidebar.write("4. Click 'Clear' to redraw.")

# Main title
st.title("✍️ Handwritten Digit Recognition")

# Create a container for layout
col1, col2 = st.columns([2, 2])

with col1:
    # Add some space above the drawing canvas to align it
    st.markdown("<br>", unsafe_allow_html=True)  # Adds vertical space
    st.subheader("Write the Digit")
    # Set up the drawing canvas
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Button to clear canvas
    if st.button("Clear Drawing"):
        st.session_state["canvas_key"] = "canvas_" + str(np.random.randint(1000))  # Update the key
        st.rerun()  # Forces a rerun of the script to clear the canvas


with col2:
    if canvas_result.image_data is not None and np.any(canvas_result.image_data):
        # Convert image to grayscale
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)

        # Invert colors (to match MNIST)
        img = cv2.bitwise_not(img)

        # Apply threshold to remove noise
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # Resize to 28x28 pixels (same as MNIST)
        img = cv2.resize(img, (28, 28))

        # Normalize (convert pixel values from 0-255 to 0-1)
        img = img.astype('float32') / 255.0

        # Expand dimensions to match model input shape (1, 28, 28, 1)
        img = np.expand_dims(img, axis=-1)  # (28, 28, 1)
        img = np.expand_dims(img, axis=0)   # (1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Display the prediction
        st.subheader(f"🧠 Predicted Digit: **{predicted_class}**")

        # Show confidence scores in a bar chart
        fig, ax = plt.subplots()
        ax.bar(range(10), prediction[0], color="blue")
        ax.set_xticks(range(10))
        ax.set_xlabel("Digits")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)
    else:
        st.subheader("✍️ Please draw a digit before predicting.")

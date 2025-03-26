import os
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2

# Load the trained models (replace with correct paths)
digit_model = tf.keras.models.load_model('digit_model.keras')
symbol_model = tf.keras.models.load_model('symbol_model.h5')

# Class names for symbols
symbol_class_names = ['+', '-', '*', '/']

# Streamlit UI setup
st.set_page_config(page_title="Handwritten Digit & Symbol Recognition", layout="wide")
st.title("✍️ Handwritten Digit & Symbol Recognition + Calculation")

# Layout for UI components
col1, col2, col3 = st.columns([1, 1, 1])  # Equal-sized columns

# For the first digit input
with col1:
    st.subheader("Draw the First Digit")
    canvas_result_digit1 = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_digit1"
    )

# For the symbol input
with col2:
    st.subheader("Draw a Mathematical Symbol")
    canvas_result_symbol = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_symbol"
    )

# For the second digit input
with col3:
    st.subheader("Draw the Second Digit")
    canvas_result_digit2 = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_digit2"
    )

# Function to perform calculation
def perform_calculation(digit1, symbol, digit2):
    if symbol == '+':
        return digit1 + digit2
    elif symbol == '-':
        return digit1 - digit2
    elif symbol == '*':
        return digit1 * digit2
    elif symbol == '/':
        return digit1 / digit2
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

# Check if all canvases contain drawings
if canvas_result_digit1.image_data is not None and np.any(canvas_result_digit1.image_data) and \
   canvas_result_symbol.image_data is not None and np.any(canvas_result_symbol.image_data) and \
   canvas_result_digit2.image_data is not None and np.any(canvas_result_digit2.image_data):

    # Process First Digit Image
    img_digit1 = cv2.cvtColor(canvas_result_digit1.image_data, cv2.COLOR_RGBA2GRAY)
    img_digit1 = cv2.bitwise_not(img_digit1)
    _, img_digit1 = cv2.threshold(img_digit1, 128, 255, cv2.THRESH_BINARY)
    img_digit1 = cv2.resize(img_digit1, (28, 28))
    img_digit1 = img_digit1.astype('float32') / 255.0
    img_digit1 = np.expand_dims(img_digit1, axis=-1)
    img_digit1 = np.expand_dims(img_digit1, axis=0)

    # Predict the first digit
    digit1_prediction = digit_model.predict(img_digit1)
    predicted_digit1 = np.argmax(digit1_prediction)

    # Process Symbol Image
    img_symbol = cv2.cvtColor(canvas_result_symbol.image_data, cv2.COLOR_RGBA2GRAY)
    img_symbol = cv2.bitwise_not(img_symbol)
    _, img_symbol = cv2.threshold(img_symbol, 128, 255, cv2.THRESH_BINARY)
    img_symbol = cv2.resize(img_symbol, (28, 28))
    img_symbol = img_symbol.astype('float32') / 255.0
    img_symbol = np.expand_dims(img_symbol, axis=-1)
    img_symbol = np.expand_dims(img_symbol, axis=0)

    # Predict the symbol
    symbol_prediction = symbol_model.predict(img_symbol)
    predicted_symbol = symbol_class_names[np.argmax(symbol_prediction)]

    # Process Second Digit Image
    img_digit2 = cv2.cvtColor(canvas_result_digit2.image_data, cv2.COLOR_RGBA2GRAY)
    img_digit2 = cv2.bitwise_not(img_digit2)
    _, img_digit2 = cv2.threshold(img_digit2, 128, 255, cv2.THRESH_BINARY)
    img_digit2 = cv2.resize(img_digit2, (28, 28))
    img_digit2 = img_digit2.astype('float32') / 255.0
    img_digit2 = np.expand_dims(img_digit2, axis=-1)
    img_digit2 = np.expand_dims(img_digit2, axis=0)

    # Predict the second digit
    digit2_prediction = digit_model.predict(img_digit2)
    predicted_digit2 = np.argmax(digit2_prediction)

    # --- Arrange Confidence Graphs Below Each Canvas ---
    graph_col1, graph_col2, graph_col3 = st.columns([1, 1, 1])

    # Confidence Graph for First Digit
    with graph_col1:
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
        ax.bar(range(10), digit1_prediction[0], color="blue")
        ax.set_xticks(range(10))
        ax.set_xlabel("Digits")
        ax.set_ylabel("Confidence")
        ax.set_title("First Digit Confidence", fontsize=10)
        st.pyplot(fig)

    # Confidence Graph for Symbol
    with graph_col2:
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
        ax.bar(symbol_class_names, symbol_prediction[0], color="blue")
        ax.set_xlabel("Symbols")
        ax.set_ylabel("Confidence")
        ax.set_title("Symbol Confidence", fontsize=10)
        st.pyplot(fig)

    # Confidence Graph for Second Digit
    with graph_col3:
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
        ax.bar(range(10), digit2_prediction[0], color="blue")
        ax.set_xticks(range(10))
        ax.set_xlabel("Digits")
        ax.set_ylabel("Confidence")
        ax.set_title("Second Digit Confidence", fontsize=10)
        st.pyplot(fig)


    # Display Predictions
    st.subheader(f"🧠 Predicted First Digit: **{predicted_digit1}**")
    st.subheader(f"🔢 Predicted Symbol: **{predicted_symbol}**")
    st.subheader(f"🧠 Predicted Second Digit: **{predicted_digit2}**")

    # Perform the Calculation
    try:
        result = perform_calculation(predicted_digit1, predicted_symbol, predicted_digit2)
        st.subheader(f"🧮 Calculation Result: **{result}**")
    except ValueError as e:
        st.error(f"Error: {e}")

else:
    st.subheader("✍️ Please draw the first digit, symbol, and second digit before predicting.")

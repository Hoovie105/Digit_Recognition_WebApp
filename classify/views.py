import tensorflow as tf
from tensorflow import keras
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from PIL import Image, ImageOps
import base64
from io import BytesIO
import matplotlib.pyplot as plt  # For debugging

# Load the trained model
model = keras.models.load_model("improved_cnn_digit_recognition.h5")

def preprocess_image(image):
    """Preprocesses the uploaded image for MNIST-style classification."""
    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Invert colors if needed (ensure white digit on black background)
    if np.mean(image) > 128:
        image = ImageOps.invert(image)

    # Convert to NumPy array
    img_array = np.array(image)

    # Auto-crop: Remove unnecessary white space around the digit
    nonzero_pixels = np.argwhere(img_array > 20)  # Find all non-black pixels
    if nonzero_pixels.shape[0] > 0:  
        min_row, min_col = nonzero_pixels.min(axis=0)
        max_row, max_col = nonzero_pixels.max(axis=0)
        image = image.crop((min_col, min_row, max_col, max_row))

    # Resize while maintaining aspect ratio (fit within 28x28)
    image.thumbnail((28, 28), Image.LANCZOS)

    # Create a blank 28x28 image and paste the digit in the center
    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))

    # Convert to NumPy array, normalize, and reshape for model
    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Debugging: Save processed image
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.savefig("debug_preprocessed.png")

    return img_array

def process_drawn_image(image_data):
    """Processes the drawn image from the canvas."""
    # Decode the base64 image data
    image_data = image_data.split(",")[1]  # Remove the data URL part
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes))
    return preprocess_image(img)

def classify_digit(request):
    if request.method == "POST":
        image = request.FILES.get("image")
        drawn_image = request.POST.get("drawn_image")

        try:
            # If the user uploaded an image
            if image:
                img = Image.open(image)
                processed_img = preprocess_image(img)
            # If the user drew on the canvas
            elif drawn_image:
                processed_img = process_drawn_image(drawn_image)

            # Get model prediction and confidence
            prediction = model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Convert to percentage

            return render(request, "result.html", {"digit": digit, "confidence": confidence})

        except Exception as e:
            return render(request, "upload.html", {"error": "Invalid image or drawing."})

    return render(request, "upload.html")

from flask import Flask, request, render_template
import os
import tifffile as tiff
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

MODEL_PATH = r'D:\Cellula Technologies\Week5\unet_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def normalize_image_channel(image_channel):
    channel_min = np.min(image_channel)
    channel_max = np.max(image_channel)
    
    return (image_channel - channel_min) / (channel_max - channel_min)

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)
    return file_path

def create_rgb_image(image_data):
    rgb_channels = [normalize_image_channel(image_data[:, :, i]) for i in range(1, 4)]
    return np.clip(np.stack(rgb_channels, axis=-1) * 255, 0, 255).astype(np.uint8)

def generate_overlay(rgb_image, mask):
    mask_image = np.zeros_like(rgb_image)
    mask_image[:, :, 2] = mask * 255 
    return Image.blend(Image.fromarray(rgb_image), Image.fromarray(mask_image), alpha=0.5)

def convert_image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return 'No selected file', 400

    file_path = save_uploaded_file(uploaded_file)

    image_data = tiff.imread(file_path)
    input_image = np.expand_dims(image_data, axis=0)
    mask_prediction = model.predict(input_image)
    mask = (np.squeeze(mask_prediction) > 0.5).astype(np.uint8)

    rgb_image = create_rgb_image(image_data)
    
    overlay_image = generate_overlay(rgb_image, mask)
    encoded_input_image = convert_image_to_base64(Image.fromarray(rgb_image))
    encoded_overlay_image = convert_image_to_base64(overlay_image)

    return render_template('index.html', rgb_image=encoded_input_image, output_image=encoded_overlay_image)

if __name__ == '__main__':
    app.run(debug=True)

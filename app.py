import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gdown

# --- CONFIG ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CONFIDENCE_THRESHOLD = 0.70

# --- APP SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- UTILITIES ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- GOOGLE DRIVE FILE IDS ---
DRIVE_MODELS = {
    'densenet_trained_model.keras': '1_irHVLn-OqNSHlq1Hn5Km8Bj7wSU8_5i',
    'best_resnet50_model.keras': '1RYLk_pZ9EZTcSMjUDLv7Fzn3hnw_XRmg',
    'mobilenetv2_trained_model.keras': '1hoZZ7OO2yxOmYG6EQ73MA05mSsZLLWUT'
}

def download_from_drive(filename, file_id):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# --- DOWNLOAD MODELS FROM DRIVE IF NOT EXISTS ---
for model_filename, file_id in DRIVE_MODELS.items():
    download_from_drive(model_filename, file_id)

# --- LOAD MODELS & CLASS INDICES ---
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

MODELS = {
    'densenet': {
        'model': tf.keras.models.load_model('densenet_trained_model.keras'),
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'input_size': (224, 224)
    },
    'mobilenetv2': {
        'model': tf.keras.models.load_model('mobilenetv2_trained_model.keras'),
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'input_size': (224, 224)
    },
    'resnet50': {
        'model': tf.keras.models.load_model('best_resnet50_model.keras'),
        'preprocess': tf.keras.applications.resnet50.preprocess_input,
        'input_size': (224, 224)
    }
}

# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    confidence = None
    warning = None
    filename = None
    selected_model = None

    if request.method == 'POST':
        selected_model = request.form.get('model')
        model_data = MODELS.get(selected_model)

        if not model_data:
            warning = "Invalid model selection."
            return render_template('index.html', warning=warning)

        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            warning = "Please upload a valid image."
            return render_template('index.html', warning=warning)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Preprocess image dynamically based on model
        input_size = model_data['input_size']
        img = image.load_img(filepath, target_size=input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = model_data['preprocess'](x)

        # Predict
        preds = model_data['model'].predict(x)[0]
        top_idx = np.argmax(preds)
        top_conf = preds[top_idx]

        if top_conf < CONFIDENCE_THRESHOLD:
            warning = "Prediction confidence too low. Please try another image."
        else:
            pred_class = idx_to_class[top_idx]
            confidence = f"{top_conf * 100:.2f}%"

    # Sort models alphabetically for dropdown
    sorted_models = dict(sorted(MODELS.items()))

    return render_template('index.html',
                           filename=filename,
                           pred_class=pred_class,
                           confidence=confidence,
                           warning=warning,
                           selected_model=selected_model,
                           models=sorted_models.keys())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

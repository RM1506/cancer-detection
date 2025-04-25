import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CONFIDENCE_THRESHOLD = 0.70

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

short_to_full = {
    "colon_aca": "Colon adenocarcinoma",
    "colon_n": "Colon benign tissue",
    "lung_aca": "Lung adenocarcinoma",
    "lung_n": "Lung benign tissue",
    "lung_scc": "Lung squamous cell carcinoma"
}

# Models with human-readable names
model_definitions = {
    'alexnet': {
        'name': 'AlexNet',
        'model': tf.keras.models.load_model('alexnet_best_model.keras'),
        'preprocess': lambda x: x / 255.0,
        'input_size': (227, 227)
    },
    'densenet': {
        'name': 'DenseNet121',
        'model': tf.keras.models.load_model('densenet121_best_model.keras'),
        'preprocess': tf.keras.applications.densenet.preprocess_input,
        'input_size': (224, 224)
    },
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'model': tf.keras.models.load_model('mobilenetv2_best_model.keras'),
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'input_size': (224, 224)
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    confidence = None
    warning = None
    filename = None
    selected_model = None

    if request.method == 'POST':
        selected_model = request.form.get('model')
        model_data = model_definitions.get(selected_model)

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

        input_size = model_data['input_size']
        img = image.load_img(filepath, target_size=input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = model_data['preprocess'](x)

        preds = model_data['model'].predict(x)[0]
        top_idx = np.argmax(preds)
        top_conf = preds[top_idx]

        if top_conf < CONFIDENCE_THRESHOLD:
            warning = "Prediction confidence too low. Please try another image."
        else:
            pred_class_short = idx_to_class[top_idx]
            pred_class = short_to_full.get(pred_class_short, pred_class_short)
            confidence = f"{top_conf * 100:.2f}%"

    # Just pass model_key => readable_name to HTML
    model_display_names = {key: data['name'] for key, data in model_definitions.items()}

    return render_template('index.html',
                           filename=filename,
                           pred_class=pred_class,
                           confidence=confidence,
                           warning=warning,
                           selected_model=selected_model,
                           models=model_display_names)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, redirect , render_template
import os
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import uuid
import json
from pathlib import Path
import sys

regression_model = pickle.load(open('model.pkl', 'rb'))





app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret')

# configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model
model = VGG16()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['GET'])
def hello():
    return render_template('image_classifier.html')

@app.route("/predict",methods=['POST'])

def predict():
    if 'imagefile' not in request.files:
        flash('No file part')
        return render_template('image_classifier.html')

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        flash('No selected file')
        return render_template('image_classifier.html')

    if not allowed_file(imagefile.filename):
        flash('File type not allowed')
        return render_template('image_classifier.html')

    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    return render_template("image_classifier.html", prediction=classification)


@app.route("/regpredict")
def hello_predict():
    return render_template('reg.html')

@app.route("/regpredict",methods=['POST'])

def regpredict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regression_model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('reg.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = regression_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


CORS(app)  # Allow frontend JS to call API if needed

LOCAL_GPT_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "llama-3-8b-gpt-4o-ru1.0"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Serve HTML page
@app.route('/textgen')
def index():
    return render_template('llm.html')

# GPT chat API
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing messages"}), 400

    payload = {
        "model": MODEL_NAME,
        "messages": data["messages"],
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        r = requests.post(LOCAL_GPT_URL, json=payload, timeout=120)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Create a new chat and return its id
@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    chat_id = uuid.uuid4().hex
    file = DATA_DIR / f"{chat_id}.json"
    file.write_text(json.dumps({"messages": []}, indent=2))
    return jsonify({"chat_id": chat_id})


# Save messages for a chat
@app.route('/api/save', methods=['POST'])
def save_chat():
    data = request.get_json() or {}
    chat_id = data.get('chat_id')
    messages = data.get('messages')
    if not chat_id or messages is None:
        return jsonify({"error": "chat_id and messages required"}), 400
    file = DATA_DIR / f"{chat_id}.json"
    file.write_text(json.dumps({"messages": messages}, indent=2))
    return jsonify({"status": "ok"})


# Load messages for a chat
@app.route('/api/load', methods=['POST'])
def load_chat():
    data = request.get_json() or {}
    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({"error": "chat_id required"}), 400
    file = DATA_DIR / f"{chat_id}.json"
    if not file.exists():
        return jsonify({"error": "not_found"}), 404
    content = json.loads(file.read_text())
    return jsonify(content)


# Delete a message by index in a chat
@app.route('/api/delete_message', methods=['POST'])
def delete_message():
    data = request.get_json() or {}
    chat_id = data.get('chat_id')
    index = data.get('index')
    if not chat_id or index is None:
        return jsonify({"error": "chat_id and index required"}), 400
    file = DATA_DIR / f"{chat_id}.json"
    if not file.exists():
        return jsonify({"error": "not_found"}), 404
    content = json.loads(file.read_text())
    messages = content.get('messages', [])
    try:
        idx = int(index)
        if idx < 0 or idx >= len(messages):
            raise IndexError
        messages.pop(idx)
    except (ValueError, IndexError):
        return jsonify({"error": "invalid_index"}), 400
    file.write_text(json.dumps({"messages": messages}, indent=2))
    return jsonify({"status": "ok", "messages": messages})
@app.route("/")
@app.route("/home")

def home():
    return render_template("home.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors


import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('models/model_checkpoint13.keras') 

class_labels=['Clams','Corals','Crabs','Dolphin','Eel','Fish','Jelly','Lobster','Nudibranchs','Octopus','Otter','Penguin','Puffers','Sea_Rays','Sea_Urchins','Seahorse','Seal','Sharks','Shrimps','Squid','Starfish','Turtle','Whale']  # replace with your class names

def preprocess_image(img: Image.Image):
    img = img.resize((256, 256))  
    img = img.convert('RGB')
    img=np.expand_dims(img, axis=0)
    img_array = np.array(img)
    return img_array

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    
    results = []
    
    for file in files:
        try:
            img = Image.open(file.stream)

            # Preprocess the image
            processed_image = preprocess_image(img)

            # Predict the class of the image
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)

            # Append each result
            results.append({
                'filename': file.filename,
                'predicted_class': class_labels[predicted_class],
                'confidence': float(confidence)
            })
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return render_template('prediction.html',results=results) 

if __name__=="__main__":
    app.run(threaded=True)
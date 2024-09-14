import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# Load the model
model = load_model('models/model_checkpoint13.keras') 

class_labels = ['Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers', 'Sea_Rays', 'Sea_Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimps', 'Squid', 'Starfish', 'Turtle', 'Whale']  # replace with your class names

def preprocess_image(img: Image.Image):
    img = img.resize((256, 256))  
    img = img.convert('RGB')
    img_array = np.expand_dims(np.array(img), axis=0)
    return img_array

st.title('Image Classification App')

uploaded_files = st.file_uploader("Choose images", type=['jpg', 'png'], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            processed_image = preprocess_image(img)
            
            # Predict the class of the image
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            results.append({
                'filename': uploaded_file.name,
                'predicted_class': class_labels[predicted_class],
                'confidence':confidence
            })
            
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'error': str(e)
            })
    
    for result in results:
        st.write(f"Filename: {result['filename']}")
        if 'error' in result:
            st.write(f"Error: {result['error']}")
        else:
            st.write(f"Predicted Class: {result['predicted_class']}")
            st.write(f"Confidence: {result['confidence']:.2f}")


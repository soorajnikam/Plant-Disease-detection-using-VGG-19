from flask import Flask, request, jsonify
#from tf import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
#from keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)


model = load_model("plant_disease_model.h5")

@app.route('/predict', methods=["POST"])
def predict():
    file = request.files['file']
    file.save(file.filename)
    img = image.load_img(file.filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    img_array /= 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Map predicted class index to class label
    class_labels = {'Target Spot': 0,
                'YellowLeaf Curl Virus': 1,
                'Healthy': 2}
    predicted_label = [k for k, v in class_labels.items() if v == predicted_class][0]

    print("Predicted class:", predicted_label)
    return jsonify({'result': predicted_label}), 200

@app.route('/')
def index():    
    return "HELLO!!!"

app.run(port=6699, debug=True)
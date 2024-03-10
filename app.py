import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS  # Import CORS from flask_cors
from model_predictor import predict_image
from model_predictor import preprocess_image
import json
#import socket
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by passing the Flask app instance

# Define the directory to save uploaded images
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    print('Request for index page received')
    # read the data from data.json file
    with open('data.json') as json_file:
        data = json.load(json_file)
        # see if there is any property called webHitsCount
        if 'webHitsCount' in data:
            # if yes, increment the count
            data['webHitsCount'] += 1
    # write the updated data to the file
    with open('data.json', 'w') as json_file:
        json.dump(data, json_file)
    # return the index.html page
        
    return render_template('index.html', webHitsCount=data['webHitsCount'])

@app.route('/RetinaAPI/v1/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

@app.route('/RetinaAPI/v1/preprocess', methods=['POST'])
def preprocess():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    
    # Use the preprocess_image function from model_predictor.py
    preprocessed_image_path = preprocess_image(file)
    # print the results
    print('🔴 > Preprocessed image:', preprocessed_image_path)

    # Return the preprocessed image
    return send_file(preprocessed_image_path, mimetype='image/jpg')

@app.route('/RetinaAPI/v1/predict', methods=['GET'])
def predict():
    predicted_class, confidence, predictions = predict_image()
    # print the results
    print('🔴 > Predicted class:', predicted_class)
    print('🔴 > Confidence:', confidence)
    print('🔴 > Predictions:', predictions)

    # Return the predictions in JSON format
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence, 'predictions': predictions})

if __name__ == '__main__':
    #____with in network____
    # host = socket.gethostbyname(socket.gethostname())
    # port = 5000
    # print(f"Server is listening at 👉👉: http://{host}:{port}")
    # app.run(host=host, port=port,debug=True)

    # ____local____
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

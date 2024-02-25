import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from model_predictor import predict_image
#import socket
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by passing the Flask app instance

# Define the directory to save uploaded images
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/RetinaAPI/v1/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

@app.route('/RetinaAPI/v1/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    
    # Use the predict_image function from model_predictor.py
    predicted_class, confidence, predictions = predict_image(file)
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

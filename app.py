import os
from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_cors import CORS
from model_predictor import predict_image, preprocess_image, load_model_if_needed, init
import json
import google.generativeai as genai
from flask import jsonify, request
import time
from bs4 import BeautifulSoup


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    print('Request for index page received')
    with open('data.json') as json_file:
        data = json.load(json_file)
        if 'webHitsCount' in data:
            data['webHitsCount'] += 1
    with open('data.json', 'w') as json_file:
        json.dump(data, json_file)
    return render_template('index.html', webHitsCount=data['webHitsCount'])

@app.route('/RetinaAPI/v1/ping', methods=['GET'])
def ping():
    return jsonify({'response': 'pong'})

# Configure your API key
genai.configure(api_key="AIzaSyC5rVEzNVCW6HDeQXO0R7wtqwFt6_ETB3E")
# Choose a model that's appropriate for your use case
model = genai.GenerativeModel('gemini-1.5-flash')



@app.route('/RetinaAPI/v1/genai', methods=['POST'])
def generateContent():
    try:
        # Extract class name from the request body
        data = request.get_json()
        className = data.get('className')
        print("Class Name:", className)
        if not className:
            return jsonify({"error": "className is required"}), 400

        # Define the prompt for the AI model
        prompt = f"""
            Generate content for diabetic retinopathy stage '{className}' with the following structure:
            <response>
                <description>content of description</description>
                <details>
                    <short_description>content of short description</short_description>
                    <stage>content of stage</stage>
                    <precautions>content of precautions</precautions>
                </details>
            </response>
        """

        # Generate content using the AI model (replace this with your actual model call)
        response = model.generate_content(prompt)  # Placeholder for actual AI model call
        print("Raw Response:", response)

        # Access the specific class details from the response
        extracted_info = response.candidates[0].content.parts[0].text
        print("Extracted Response:", extracted_info)

        # Parse the response using BeautifulSoup
        soup = BeautifulSoup(extracted_info, 'html.parser')
        description = soup.find('description').text if soup.find('description') else ''
        short_description = soup.find('short_description').text if soup.find('short_description') else ''
        stage = soup.find('stage').text if soup.find('stage') else ''
        precautions = soup.find('precautions').text if soup.find('precautions') else ''

        # Construct the response in the desired format
        formatted_response = {
            className: {
                'description': className,
                'details': {
                    'short_description': short_description,
                    'stage': stage,
                    'precautions': precautions
                }
            }
        }

        print("Formatted Response:", formatted_response)
        return jsonify(formatted_response), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/RetinaAPI/v1/preprocess', methods=['POST'])
def preprocess():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    preprocessed_image_path = preprocess_image(file)
    print('🔴 > Preprocessed image:', preprocessed_image_path)
    return send_file(preprocessed_image_path, mimetype='image/jpg')

@app.route('/RetinaAPI/v1/predict', methods=['GET'])
def predict():
    predicted_class, confidence, predictions = predict_image()
    print('🔴 > Predicted class:', predicted_class)
    print('🔴 > Confidence:', confidence)
    print('🔴 > Predictions:', predictions)
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence, 'predictions': predictions})


@app.route('/RetinaAPI/v1/xai', methods=['GET'])
def predict_xai():
    xaiimg = init()  # Initialize or get the image path
    response = make_response(send_file(xaiimg, mimetype='image/jpg'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    load_model_if_needed()  # Ensure the model is loaded once at startup
    app.run(host="0.0.0.0", port=5000, debug=True)

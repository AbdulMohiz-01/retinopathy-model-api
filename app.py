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
            Generate a concise and precise content for diabetic retinopathy stage '{className}' with the following structure:
            <response>
                <description>Short and precise description of {className}</description>
                <details>
                    <short_description>Brief summary of the stage {className}</short_description>
                    <stage>Detailed description of the stage {className}</stage>
                    <precautions>Key precautions and necessary actions for {className}</precautions>
                </details>
            </response>
            Ensure that the response is accurate and factual. If unable to generate the correct text, respond with 'CAN NOT GENERATE'.

            Here are examples for few stages of diabetic retinopathy, you can take these examples as a reference to generate the content for the requested stage:
            
            Example for "No Diabetic Retinopathy":
            <response>
                <description>No Diabetic Retinopathy</description>
                <details>
                    <short_description>No signs of diabetic retinopathy.</short_description>
                    <stage>Your eyes are healthy with no signs of damage from diabetes.</stage>
                    <precautions>It's important to continue regular eye check-ups with your doctor to make sure your eyes stay healthy.</precautions>
                </details>
            </response>

            Example for "Proliferative Diabetic Retinopathy":
            <response>
                <description>Proliferative Diabetic Retinopathy</description>
                <details>
                    <short_description>Most severe stage of diabetic retinopathy.</short_description>
                    <stage>New abnormal blood vessels have formed in your eyes due to diabetes, which can lead to serious vision problems.</stage>
                    <precautions>Immediate medical intervention is necessary to protect your vision. Regular eye exams are crucial, and your doctor may recommend treatments like laser surgery or injections to prevent vision loss and preserve your eyesight.</precautions>
                </details>
            </response>
        """

        # Generate content using the AI model (replace this with your actual model call)
        response = model.generate_content(prompt)  # Placeholder for actual AI model call
        print("Raw Response:", response)

        # Access the specific class details from the response
        extracted_info = response.candidates[0].content.parts[0].text
        print("Extracted Response:", extracted_info)

        # Check if the response contains 'CAN NOT GENERATE'
        if 'CAN NOT GENERATE' in extracted_info:
            return jsonify({"error": "CAN NOT GENERATE"}), 400

        # Parse the response using BeautifulSoup
        soup = BeautifulSoup(extracted_info, 'html.parser')
        description = soup.find('description').text if soup.find('description') else 'CAN NOT GENERATE'
        short_description = soup.find('short_description').text if soup.find('short_description') else 'CAN NOT GENERATE'
        stage = soup.find('stage').text if soup.find('stage') else 'CAN NOT GENERATE'
        precautions = soup.find('precautions').text if soup.find('precautions') else 'CAN NOT GENERATE'

        # Construct the response in the desired format
        formatted_response = {
            className: {
                'description': description,
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

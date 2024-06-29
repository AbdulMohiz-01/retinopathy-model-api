import os
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from model_predictor import predict_image, preprocess_image, load_model_if_needed, init
import json
import google.generativeai as genai
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

        print(className)

        if not className:
            return jsonify({"error": "className is required"}), 400

        # Define the prompt for the AI model
        prompt = f"""
        Generate content for diabetic retinopathy stage '{className}' with the following structure:
        {{
          '{className}': {{
            'description': '',
            'details': {{
              'short_description': '',
              'stage': '',
              'precautions': ''
            }}
          }}
        }}
        """

        # Generate content using the AI model
        response = model.generate_content(prompt)
        
        # Assuming response.text contains the generated JSON
        generated_content = response.text

        # Convert the generated content to JSON
        content = eval(generated_content)  # Using eval for simplicity; consider safer parsing in production

        print(content)

        return jsonify(content), 200

    except Exception as e:
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
    xaiimg = init()
    return send_file(xaiimg, mimetype='image/jpg')

if __name__ == '__main__':
    load_model_if_needed()  # Ensure the model is loaded once at startup
    app.run(host="0.0.0.0", port=5000, debug=True)

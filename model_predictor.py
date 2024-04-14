import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

image_path = 'uploads/image.jpg'


def preprocess_image(image_file):
     # save this image file to the upload folder
    image_file.save('uploads/image.jpg')

    # get the image path
    image_path = 'uploads/image.jpg'

    resize_images('uploads/')

    img = load_ben_color(image_path)

    # save the processed image
    cv2.imwrite('uploads/image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return 'uploads/image.jpg'



# Define a global variable to hold the loaded model
loaded_model = None
loaded_base_model = None

def load_model_if_needed():
    global loaded_model
    global loaded_base_model
    if loaded_model is None:
        # Load the model if it hasn't been loaded yet
        # call model_structure() and get base model and model from it
        loaded_model , loaded_base_model = model_structure()
        # artifact\model_weights.h5
        loaded_model.load_weights("./artifact/model_weights.h5")
        loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

        # print("done loading model")
        # loaded_model = tf.keras.models.load_model('./artifact/EfficientNetB0DR_96.h5')

def model_structure():
    # Create Model Structure
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = 6

    # create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
    # we will use efficientnetb3 from EfficientNet family.
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(2040, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                    bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])

    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    model.summary()
    return model, base_model


def predict_image():
    global loaded_model
    # Load the model if needed
    load_model_if_needed()
    
    image_path = 'uploads/image.jpg'
    # Load and preprocess the image for prediction
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make predictions using the loaded model
    predictions = loaded_model.predict(img_array)

    # Assuming you have a classification model, you might want to decode the predictions
    class_labels = ['0', '1', '2', '3', '4','5']  # Replace with your actual class labels
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_labels[predicted_class_index]
    # run grad cam code
    # run()
    # Return the predictions
    return predicted_class_name, float(predictions[0][predicted_class_index]), predictions[0].tolist()



def resize_images(output_folder_path):
    for root, dirs, files in os.walk(output_folder_path):
        for file in files:
            # Check if the file is an image (you may want to add more image extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                try:
                    # Open the image file
                    img = Image.open(file_path)
                    # Resize the image to 300 x 300
                    resized_img = img.resize((224, 224))
                    # Save the resized image
                    resized_img.save(file_path)
                    # # Remove the original photo
                    # os.remove(file_path)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")



def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    

def load_ben_color(path, sigmaX=10):
    """
    Load image, crop out black borders, and enhance contrast
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

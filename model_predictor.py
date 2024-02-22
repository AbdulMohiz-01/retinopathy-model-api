import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import image

def predict_image(image_file):

    # save this image file to the upload folder
    image_file.save('uploads/image.jpg')

    # get the image path
    image_path = 'uploads/image.jpg'

    resize_images('uploads/')

    img = load_ben_color(image_path)

    # save the processed image
    cv2.imwrite('uploads/image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))






    # Load and preprocess the image for prediction
    # img = Image.open(BytesIO(image_file.read()))
    # img = img.resize((224, 224))
    # Preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    # img_array = img_array.astype('float32') / 255.0  # Normalize the image data

    # load the model
    model = tf.keras.models.load_model('./artifact/retina.h5')
    # Make predictions
    predictions = model.predict(img_array)

    # Assuming you have a classification model, you might want to decode the predictions
    class_labels = ['0', '1', '2', '3', '4', '5']  # Replace with your actual class labels
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_labels[predicted_class_index]

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

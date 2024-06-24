# model_predictor.py

import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
import keras

image_path = 'uploads/image.jpg'

# Define global variables to hold the loaded models
loaded_model = None
loaded_base_model = None

def load_model_if_needed():
    global loaded_model, loaded_base_model
    if loaded_model is None:
        loaded_model, loaded_base_model = model_structure()
        loaded_model.load_weights("./artifact/model_weights.h5")

def model_structure():
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = 6

    base_model = EfficientNetB3(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(2040, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(class_count, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model, base_model

def preprocess_image(image_file):
    image_file.save('uploads/image.jpg')
    image_path = 'uploads/image.jpg'
    resize_images('uploads/')
    img = load_ben_color(image_path)
    cv2.imwrite('uploads/image.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return 'uploads/image.jpg'

def predict_image():
    global loaded_model
    load_model_if_needed()
    
    image_path = 'uploads/image.jpg'
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = loaded_model.predict(img_array)
    class_labels = ['0', '1', '2', '3', '4', '5']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_labels[predicted_class_index]

    return predicted_class_name, float(predictions[0][predicted_class_index]), predictions[0].tolist()

def resize_images(output_folder_path):
    for root, dirs, files in os.walk(output_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    resized_img = img.resize((224, 224))
                    resized_img.save(file_path)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

def crop_image_from_gray(img, tol=7):
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
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def init():
    global loaded_model, loaded_base_model
    load_model_if_needed()  # Ensure models are loaded

    if loaded_model is None or loaded_base_model is None:
        raise Exception("Model or base model is not loaded properly.")

    img_path = 'uploads/image.jpg'
    model_builder = loaded_base_model
    img_size = (224, 224)
    last_conv_layer_name = "top_conv"

    original_image_path = img_path
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    img_array = preprocess_image_for_xai(img_path, img_size)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    class_index = np.argmax(loaded_model.predict(img_array))
    grads = get_gradient(img_tensor, loaded_model, class_index)
    heatmap = generate_gradcam(img_tensor, loaded_base_model, last_conv_layer_name, class_index)
    overlayed_image = overlay_heatmap(heatmap, original_image)

    cv2.imwrite('uploads/image_xai.jpg', cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    return 'uploads/image_xai.jpg'

def preprocess_image_for_xai(img_path, img_size):
    img = keras.utils.load_img(img_path, target_size=img_size)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def get_gradient(img_array, model, class_index):
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        preds = model(img_array)
        class_output = preds[:, class_index]
    grads = tape.gradient(class_output, img_array)
    return grads

def generate_gradcam(img_array, model, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_output = preds[:, class_index]
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return tf.reshape(heatmap, (heatmap.shape[1], heatmap.shape[2])).numpy()

def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(heatmap, 0.5, original_image, 0.5, 0)
    return overlayed_img

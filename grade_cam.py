import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2
from model_predictor import loaded_base_model ,loaded_model
from model_predictor import model_structure
from tensorflow.keras.optimizers import Adamax


model=loaded_base_model
base_model = loaded_base_model
# base_model = loaded_base_model
# model = loaded_model


# model_weights.h5
model , base_model = model_structure()
# model.load_weights("./artifact/DR_model_15_19 (94.660).h5")
model.load_weights("./artifact/model_weights.h5")
# model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])



img_path = 'uploads/image.jpg'

# Assuming base_model and img_size are defined elsewhere
model_builder = loaded_base_model
img_size = (224, 224)
# last_conv_layer_name = "top_conv"
# img_path = "/content/flipped_both_processed_11896_left.jpeg"

# Function to preprocess the image for the model
def preprocess_image(img_path, img_size):
    img = keras.utils.load_img(img_path, target_size=img_size)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to get the gradients of the class with respect to the input image
def get_gradient(img_array, model, class_index):
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        preds = model(img_array)
        class_output = preds[:, class_index]
    grads = tape.gradient(class_output, img_array)
    return grads

# Function to generate the Grad-CAM heatmap
def generate_gradcam(img_array, model, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [base_model.inputs], [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_output = preds[:, class_index]
        print(class_index)
        print("Predicted Class Label:", class_output.numpy())
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return tf.reshape(heatmap, (heatmap.shape[1], heatmap.shape[2])).numpy()

    # Function to overlay heatmap on the original image
def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(heatmap, 0.5, original_image, 0.5, 0)
    return overlayed_img



# for i in base_model.layers:
#   if i.name.endswith("expand_conv"):

# # last_conv_layer_name = "top_conv"   
# last_conv_layer_name = "block7b_expand_conv" 
# # last_conv_layer_name = i.name
# # img_path = "/content/content/content/DR_dataset/4/flipped_both_processed_1084_left.jpeg"
# # Load the original image
# original_image_path = img_path
# original_image = cv2.imread(original_image_path)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# # Preprocess the image for the model
# img_array = preprocess_image(img_path, img_size)

# # Convert the input image to a TensorFlow tensor
# img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

# # Compute the gradients of the class with respect to the input image
# class_index = np.argmax(model.predict(img_array))
# grads = get_gradient(img_tensor, model, class_index)

# # Generate Grad-CAM heatmap
# heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name, class_index)

def init():
    global grads
    global heatmap
    # last_conv_layer_name = "block7b_expand_conv" 
    last_conv_layer_name = "top_conv" 
    original_image_path = img_path
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Preprocess the image for the model
    img_array = preprocess_image(img_path, img_size)

    # Convert the input image to a TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Compute the gradients of the class with respect to the input image
    class_index = np.argmax(model.predict(img_array))
    grads = get_gradient(img_tensor, model, class_index)

    
    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(img_tensor, model, last_conv_layer_name, class_index)
    # Visualize the heatmap
    # plt.imshow(heatmap, cmap='viridis')
    # plt.axis('off')
    # plt.show()

    # Overlay heatmap on the original image
    overlayed_image = overlay_heatmap(heatmap, original_image)

    # # Display the overlayed image
    # plt.imshow(overlayed_image)
    # plt.axis('off')
    # plt.show()
    cv2.imwrite('uploads/image_xai.jpg', cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    return 'uploads/image_xai.jpg'
    # return overlayed_image

# # Overlay heatmap on the original image
# overlayed_image = overlay_heatmap(heatmap, original_image)

# # Display the overlayed image
# plt.imshow(overlayed_image)
# plt.axis('off')
# plt.show()



# # Visualize the heatmap
# plt.imshow(heatmap, cmap='viridis')
# plt.axis('off')
# # plt.show()

# # Overlay heatmap on the original image
# overlayed_image = overlay_heatmap(heatmap, original_image)

# # Display the overlayed image
# plt.imshow(overlayed_image)
# plt.axis('off')
# plt.show()
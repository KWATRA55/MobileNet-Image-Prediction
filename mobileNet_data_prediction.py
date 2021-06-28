import numpy as np
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf


# download the mobilenet function 
mobile = tf.keras.applications.mobilenet.MobileNet()

# create a function to preprocess the image to required size and dimension
def prepare_image(file):
    img_path = "C:/data/cats_vs_dogs/test/dog/"  # this is your image path
    img = image.load_img(img_path + file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expand_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expand_dims)


preprocessed_image = prepare_image("126.jpg") # give your image name
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)



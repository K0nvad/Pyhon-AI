import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# defining image size
img_height = 244
img_width = 244
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # diseases

# Loading model
model = tf.keras.models.load_model('mri_disease_classifier.h5')

def predict_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width)) # loading the picture
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array) # calling the model
    predicted_class = class_labels[np.argmax(predictions)] # getting the results
    confidence = np.max(predictions) * 100 # turning data into % value

    print(f"Predicted Disease: {predicted_class} ({confidence:.2f}%)")

if __name__ == "__main__":
    print("Please provide the path to your MRI scan (e.g. 'test_image.jpg'):")
    img_path = input("Path: ")
    predict_image(img_path)

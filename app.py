import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

model = tf.keras.models.load_model("C:\\Users\\snthi\\Downloads\\cat vs dog-20250316T131901Z-001\\cat_vs_dog_model.h5")
def preprocess_image(img):
    img = img.convert("RGB")  
    img = img.resize((128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    return img_array

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog and let the model predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img)
    prediction = model.predict(img_array)

    st.write("Raw Prediction Output:", prediction)  

    if prediction[0][0] > 0.5:
        st.success("### 🐶 This is a Dog!")
    else:
        st.success("### 🐱 This is a Cat!")

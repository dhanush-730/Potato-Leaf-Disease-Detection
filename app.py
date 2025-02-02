import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1v5Ffdp0zb7LU04a6hJxq2YEVtAth-7iX"
url = 'https://drive.google.com/file/d/1v5Ffdp0zb7LU04a6hJxq2YEVtAth-7iX'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)


model_path = "trained_plant_disease_model.keras"
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 


st.sidebar.title("Potato Leaf Disease Detection Using CNN for Accurate Diagnosis")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])


from PIL import Image
img = Image.open("logo.png")

st.image(img)


if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection Using CNN for Accurate Diagnosis", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Potato Leaf Disease Detection Using CNN for Accurate Diagnosis")
    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        if(st.button("Show Image")):
            st.image(test_image,width=4,use_container_width=True)

        #Predict button
        if(st.button("Predict")):
            st.balloons()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        

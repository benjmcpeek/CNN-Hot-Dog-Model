from operator import mod
import streamlit as st
from PIL import Image
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import EarlyStopping
# import pickle
import numpy as np



st.title('Classification on Images of Hot Dogs')
st.write("""
Our app uses a Classification Neural Network model 
to decipher whether or not an image contains a hot dog or not. 
Please upload a 'jpg' or 'png' image of your choosing below to have the model determine 
if your image includes a hot dog or not!""")


image = Image.open('./hotdog_nothotdog/hotdog/1502.jpg')
st.image(image, caption= 'Example of image with hotdog(s) taken from test set of model')

image = Image.open('./hotdog_nothotdog/nothotdog/164.jpg')
st.image(image, caption= 'Example of image without hotdog(s) taken from test set of model')

img_data = st.file_uploader(label='Load an image:', type = ['jpg', 'png'])
if img_data is not None:
    uploaded_img = Image.open(img_data)
    new_img = uploaded_img.resize((299, 299))
    other_new_img = np.expand_dims(new_img, axis = 0)
    st.image(uploaded_img)
    model = load_model('./models/model_3.h5')
    if model.predict(other_new_img)[0][0] >= 0.5:
        st.write('Your image IS NOT hot dog!')
    else:
        st.write('Your image IS a hot dog!')
    # st.write(f'Your model has a loss function value of {res_3.history['val_loss'][-1]}')
    # st.write(f'Your model has an accuracy score of {res_3.history['val_acc'][-1]}')

    
# code above found at https://discuss.streamlit.io/t/drag-and-drop-image/43144
# st.write(uploaded_img)



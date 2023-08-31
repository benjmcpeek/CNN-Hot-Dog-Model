from operator import mod
import streamlit as st
import pickle

st.title('Classification on Images of Hot Dogs')
st.write("""
Our app uses a Classification Neural Network model 
to decipher whether or not an image contains a hot dog or not. 
Please upload a 'jpg' or 'png' image of your choosing below to have the model determine 
if your image includes a hot dog or not!""")

# Below code copied from 9.06 lesson GA

def load_model():
  with open('./models_copy/model_base.pkl', 'rb') as f:
    the_model = pickle.load(f)
  return the_model

model = load_model()


image = Image.open('1502.jpg')
st.image(image, caption= 'Example of image with hotdog(s) taken from test set of model')

image = Image.open('164.jpg')
st.image(image, caption= 'Example of image without hotdog(s) taken from test set of model')

img_data = st.file_uploader(label='Load an image:', type = ['jpg', 'png'])
if img_data is not None:
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img)

# code above found at https://discuss.streamlit.io/t/drag-and-drop-image/43144
st.write(uploaded_img)



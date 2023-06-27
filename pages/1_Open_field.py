import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import json
import base64
import filecmp
from urllib.request import urlopen
import time
import io
from PIL import Image as im
import cv2
import tempfile
import pathlib
from pathlib import Path
import subprocess
from io import BytesIO

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, utils


st.sidebar.image("data/logo_mlit.jpg")
st.sidebar.info(
    """
    This app is Open Source dashboard.
    """
)
st.sidebar.info("Site of MLIT JINR: "
                "[Link](https://lit.jinr.ru/).")
st.sidebar.info("The project is being created within the framework of the ML/DL/HPC ecosystem of the HybriLIT platform. Link: "
                "[here](https://hlit.jinr.ru).")
st.sidebar.image("data/logo_hlit.png")
st.sidebar.header("Dashboard")
start_fr = st.sidebar.number_input(label = "С какого кадра начать анализ?", min_value=1, max_value=100, value=25, step=1, format="%d")
st.sidebar.write('Начать анализ с ', start_fr, 'кадра')

animal = st.sidebar.selectbox('Select laboratory animal', ('Mouse', 'Rat'))
st.sidebar.write('You selected:', animal)

st.markdown("# The Open field test-system analysis")

def load_video2():
    global uploaded_file
    uploaded_file = st.file_uploader(label='Выберите видео для анализа', type=["mp4", "avi"])

    if uploaded_file is not None:
        st.markdown("## Original file ")
        st.markdown(uploaded_file.name) 
        video_file = open(uploaded_file.name, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time = 0)   
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read()) 
        
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
 
        vdata = [] 
        vidcap = cv2.VideoCapture(tfile.name) 
 
        success,image = vidcap.read() 
        count = 0 
        while success: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            vdata.append(gray) 
            success,image = vidcap.read() 
            count += 1 
        
        st.write("Количество кадров = ", count)
        vdata_arr = np.array(vdata)  

        return vdata_arr
    else:
        return None
    
vdata_arr = load_video2()
    
def get_average(frames, num=None):
    if num:
        len_frames = len(frames)
        frames = [frames[i] for i in range(0, len_frames, len_frames // num or 1)]
    return np.average(frames, axis = 0).astype(dtype=np.uint8)


def load_models():
    mod_keypoint = load_model('model/key_points_model.h5')
    return mod_keypoint
model = load_models()

def prediction(input_img, model):
    dim = (768, 768)
    X_test_median = []
    count = 0

    img = get_average(vdata_arr)
    imageTest = cv2.imread(img)

    x = np.array(imageTest)
    resized = cv2.resize(imageTest, dim, interpolation=cv2.INTER_AREA)
    X_test_median.append(resized)
    count +=1
    
    X_median= np.array(X_test_median)
    X2_medim=X_median/255
    x2 = x.reshape(1, X2_medim.shape[1], X2_medim.shape[2], 1)
    
    prediction = model.predict(x2)
    prediction_point=prediction.reshape(13,2)
    Pxy_predict = prediction_point*384 +384
    
    return prediction, Pxy_predict
    

if vdata_arr is not None:
    st.write(vdata_arr.shape)
    average_fr = get_average(vdata_arr)
    st.image(average_fr)
    
    if st.button('Analyse'):
        st.markdown('## Key points: ')
        predictions = prediction (average_fr, model)
        
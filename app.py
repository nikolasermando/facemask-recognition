# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:59:53 2022

@author: Nikolas Ermando
"""
import numpy as np
import streamlit as st
import cv2
import av
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer

# Loading Model
model = load_model("cnnmodel.h5")

# Title
st.markdown("<h1 style='text-align: center; '>Face Mask Recognition</h2>", unsafe_allow_html=True)

# Functioning Camera Button
run = st.checkbox('Use Camera')
FRAME_WINDOW = st.image([])

# Active cv2 camera and cascade frontalface classifier
camera = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4

# label prediction and color
labels_dict={0:'Mask',1:'Without Mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

# Frame processing
while run:
    rval, frame = camera.read()
    frame = cv2.flip(frame,1,1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mini = cv2.resize(frame, (frame.shape[1] // size, frame.shape[0] // size))
    faces = classifier.detectMultiScale(mini)
    
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = frame[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(64,64))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,64,64,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=result[0][0]
        if label > 0.5:
            label1 = 1
        else:
            label1 = 0
      
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label1],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label1],-1)
        cv2.putText(frame, labels_dict[label1], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    FRAME_WINDOW.image(frame)
    if run==False:
        break
    
# active camera
camera.release()

st.markdown("<h4 style='text-align: center; '>or</h2>", unsafe_allow_html=True)

# Adding another method by upload file
uploaded_files = st.file_uploader("Choose an image",type=['png','jpeg','jpg'])

# Functioning upload file processing
if uploaded_files:
    img = Image.open(uploaded_files)
    x=st.image(img)
    img = img.resize((64,64))
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
      st.markdown("<h4 style='text-align: center; '>Without Mask</h2>", unsafe_allow_html=True)
    else:
      prediction = st.markdown("<h4 style='text-align: center; '>Mask</h2>", unsafe_allow_html=True)





    
    
    

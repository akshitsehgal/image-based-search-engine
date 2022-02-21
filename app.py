from flask import Flask,render_template,request,flash,redirect
from deepface import DeepFace
import cv2
import deepface
import numpy as np
import glob
import os
import pickle
from deepface import DeepFace
import streamlit as st
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
  return render_template('index.html')

@app.route('/submit', methods=['GET','POST'])
def index():
    if request.method == 'POST':
      imagefile = request.files['imagefile']
      img_path = 'image_data/' + imagefile.filename
      imagefile.save(img_path)
      verifications = DeepFace.find(img_path ,db_path='static/images/Train',enforce_detection=False)
      paths = verifications['identity'].to_list()
      len_paths = int(0.5 * len(paths))
      paths = paths[:len_paths]
      print(paths)
      pred = True

    
    return render_template('index.html', img_path = paths)
    

if __name__ =='__main__':
  app.run(debug=False)
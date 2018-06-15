from flask import render_template
from flaskapp import app
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import os
from src.fashion_tools import image_to_feature, DeepFashion, similarity_function, rgb_image_bounding_box
import numpy as np
from keras.models import load_model
import time
import cv2

#global encoder
#encoder = load_model("models/encoder_model_current.h5")
#encoder.load_weights("models/encoder_model_weights_current.h5")
#decoder = load_model("models/decoder_model_current.h5")
#decoder.load_weights("models/decoder_model_weights_current.h5")

deepDict = DeepFashion("Top")
global deepKeys
deepKeys = [keyname for keyname in deepDict.keys()]


global allFeatures
allFeatures = np.load("models/current_feature_vector.npy")

@app.route('/')

@app.route('/index')
def index():
    imgfile = {"filepath" :  "static/images/go-away-oscar-the-grouch-t-shirt.master.png"}
    return render_template("index.html", title="Home", imgfile=imgfile)

@app.route('/mirror/<path:imgpath>')
def smart_mirror(imgpath):
    #global encoder
    fullpath_to_data = "/home/dawnis/Data/SmartMirror/DeepFashion_Data"
    encoder = load_model("models/encoder_model_current.h5")
    encoder.load_weights("models/encoder_model_weights_current.h5")
    time.sleep(3)
    #aimg = "flaskapp/static/images/bundle1/key.png"
    aimg = os.path.join("flaskapp", imgpath)
    feature_vector_main = image_to_feature(aimg, [], encoder)
    scores = [similarity_function(feature_vector_main, partner) for partner in allFeatures]
    closest = np.argsort(np.array(scores))

    for idx, x in enumerate(closest[:6]):
        keyname = deepKeys[x]
        image = rgb_image_bounding_box("/".join([fullpath_to_data, keyname]), deepDict[keyname])
        cv2.imwrite('flaskapp/static/images/match{:03d}.png'.format(idx+1), image)

    #deepKeys[closest]
    #print(feature_vector_main)
    return render_template("mirror_display.html", title="Smart Mirror App")

#flask functions .getJSON, {{_url_for...}}

@app.route('/upload')
def upload_image():
    root = Tk()
    root.update()
    root.img_path = askopenfilename(title="Choose an image")
    print(root.img_path)
    root.destroy()
    relative_path = os.path.relpath(root.img_path, "flaskapp")
    imgfile = { 'filepath' : relative_path}
    return render_template("index.html", title="Home", imgfile = imgfile )

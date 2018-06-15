from flask import render_template
from flaskapp import app
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import os
from src.fashion_tools import image_to_feature

from keras.models import load_model
import time

#global encoder
#encoder = load_model("models/encoder_model_current.h5")
#encoder.load_weights("models/encoder_model_weights_current.h5")
#decoder = load_model("models/decoder_model_current.h5")
#decoder.load_weights("models/decoder_model_weights_current.h5")

fullpath_to_data = "/home/dawnis/Data/SmartMirror/DeepFashion_Data"

@app.route('/')

@app.route('/index')
def index():

    imgfile = {"filepath" : os.path.join(app.instance_path, "/static/images/go-away-oscar-the-grouch-t-shirt.master.png")}
    return render_template("index.html", title="Home", imgfile=imgfile)

@app.route('/mirror/<imgfile.filepath>')
def smart_mirror(imgfile):
    #global encoder
    encoder = load_model("models/encoder_model_current.h5")
    encoder.load_weights("models/encoder_model_weights_current.h5")
    time.sleep(2)
    #aimg = "flaskapp/static/images/bundle1/key.png"
    aimg = imgfile.relative_path
    feature_vector_main = image_to_feature(aimg, [], encoder)
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

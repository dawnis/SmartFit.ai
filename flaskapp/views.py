from flask import render_template
from flaskapp import app
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from skimage.feature import hog
import os
from src.fashion_tools import image_to_feature, DeepFashion, similarity_function, rgb_image_bounding_box
import numpy as np
import tensorflow as tf
from keras.models import load_model
import time
import cv2

# global encoder
# decoder = load_model("models/decoder_model_current.h5")
# decoder.load_weights("models/decoder_model_weights_current.h5")

deepDict = DeepFashion("Top")
global deepKeys
deepKeys = [keyname for keyname in deepDict.keys()]

global allFeatures
allFeatures = np.load("features/current_feature_vector.npy")


#graph = tf.get_default_graph()

def my_load_model():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    global encoder
    encoder = load_model("models/encoder_model_current.h5")
    encoder.load_weights("models/encoder_model_weights_current.h5")
    global graph
    graph = tf.get_default_graph()


def encoder_predict(image_full_path):
    """
    returns full feature vector (assumes image is already bounded)
    :param image_full_path: full path to image (in Flask App)
    :return:
    """
    # global encoder
    imgraw = cv2.imread(image_full_path, 1)
    imgcrop = imgraw
    imgresize = cv2.resize(imgcrop, (128, 128))
    imgresize = imgresize / 255.
    imgresize = imgresize.astype('float32')
    with graph.as_default():
        encoded_image = encoder.predict(imgresize[None, :, :, :])
    grayscale = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
    fd = hog(grayscale, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
    fd = fd / np.max(fd)
    encoded_image = encoded_image / np.max(encoded_image)
    hsv_img = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2HSV)
    hsv_hlist = []
    num_h_elements = np.prod(hsv_img.shape[:2])
    for channel, (range, nbins) in enumerate(zip([180,255,255],[10,4,4])): #unsure if ch1 is 180 or 360
        hsv_h, bins = np.histogram(hsv_img[:,:,channel], range=(0, range), bins=nbins)
        hsv_hlist.append(hsv_h/num_h_elements)
    hsv = np.concatenate(hsv_hlist, axis=0)
    return np.concatenate((fd, hsv, encoded_image.ravel()))


@app.route('/')
@app.route('/index')
def index():
    imgfile = {"filepath": "static/images/go-away-oscar-the-grouch-t-shirt.master.png"}
    return render_template("index.html", title="Home", imgfile=imgfile)


@app.route('/mirror/<path:imgpath>')
def smart_mirror(imgpath):
    fullpath_to_data = "/home/dawnis/Data/SmartMirror/DeepFashion_Data"
    aimg = os.path.join("flaskapp", imgpath)
    imgpath_breakdown = imgpath.split(os.sep)
    imgfile = {"filepath": os.sep.join(imgpath_breakdown[1:]) }
    feature_vector_main = encoder_predict(aimg)
    scores = [similarity_function(feature_vector_main, partner) for partner in allFeatures]
    closest = np.argsort(np.array(scores))
    match = {}
    for idx, x in enumerate(closest[:4]):
        keyname = deepKeys[x]
        image = rgb_image_bounding_box("/".join([fullpath_to_data, keyname]), deepDict[keyname])
        writepath = os.path.join('flaskapp/static', keyname)
        if not os.path.exists(os.path.dirname(writepath)):
            os.makedirs(os.path.dirname(writepath))
        cv2.imwrite(writepath, image)
        match.update({"location{:02d}".format(idx+1) : keyname})
    return render_template("mirror_display.html", title="Smart Mirror App", match = match, imgfile=imgfile)


# flask functions .getJSON, {{_url_for...}}

@app.route('/upload')
def upload_image():
    root = Tk()
    root.update()
    root.img_path = askopenfilename(initialdir = 'flaskapp/static/images/', title="Choose an image")
    print(root.img_path)
    root.destroy()
    relative_path = os.path.relpath(root.img_path, "flaskapp")
    imgfile = {'filepath': relative_path}
    return render_template("index.html", title="Home", imgfile=imgfile)

from flask import render_template, request, redirect, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from shutil import copyfile
from flaskapp import app
from skimage.feature import hog
import os
from src.fashion_tools import DeepFashion, similarity_function, rgb_image_bounding_box
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2, pickle
import pdb

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


# code copied from flask.pocoo.org: uploading files (also used in upload image)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# global encoder
# decoder = load_model("models/decoder_model_current.h5")
# decoder.load_weights("models/decoder_model_weights_current.h5")

deepDict = DeepFashion("Top")
# z_img_dir = "/home/ubuntu/smartfit/flaskapp/static/z_img/womenless"
zDict = pickle.load(open("/home/ubuntu/smartfit/models/zdirectory.p", "rb"))
global deepKeys
# deepKeys = [keyname for keyname in deepDict.keys()]
deepKeys = [os.sep.join(['womenless', zDict[img]]) for img in zDict.keys()]

global allFeatures
# allFeatures = np.load("features/u_current_feature_vector.npy")
allFeatures = np.load("features/zolonda_full_feature_vectors.npy")


# graph = tf.get_default_graph()

def my_load_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    global encoder
    encoder = load_model("models/u_encoder_model_current.h5")
    encoder.load_weights("models/u_encoder_model_weights_current.h5")
    global graph
    graph = tf.get_default_graph()


def encoder_predict(image_full_path):
    """
    returns full feature vector (assumes image is already bounded)
    :param image_full_path: full path to image (in Flask App)
    :return:
    """
    # global encoder
    imgcrop = rgb_image_bounding_box(image_full_path, [],
                                     autocrop=False)  # important, autocrop True can affect performance
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
    for channel, (range, nbins) in enumerate(zip([180, 255, 255], [10, 4, 4])):  # unsure if ch1 is 180 or 360
        hsv_h, bins = np.histogram(hsv_img[:, :, channel], range=(0, range), bins=nbins)
        hsv_hlist.append(hsv_h / num_h_elements)
    hsv = np.concatenate(hsv_hlist, axis=0)
    hsv *= 10  # matches the image_to_feature function
    return np.concatenate((fd, hsv, encoded_image.ravel()))


@app.route('/')
@app.route('/index')
def index():
    imgfile = {"person": "images/model_landing.jpg", "fashion": "z_img/womenless/000102_1.jpg"}
    return render_template("index.html", title="Home", imgfile=imgfile)


@app.route('/mirror', methods=['GET'])
def smart_mirror():
    fashion = request.args.get('fashion')
    person = request.args.get('person')
    z_img_dir = "z_img"
    # z_img_dir = "deepFashion"
    aimg = os.path.join("flaskapp/static", fashion)
    vfit_base_dir = "virtual_fit"
    person_fname = person.split(os.sep)[-1][:-4]
    fashion_fname = fashion.split(os.sep)[-1][:-4]
    virtual_fit_fname = "_".join([person_fname, fashion_fname]) + ".png"
    virtual_fullpath = os.sep.join([vfit_base_dir, virtual_fit_fname])
    print(virtual_fullpath)
    if not os.path.isfile(virtual_fullpath):
        #TODO: JS TIMER
        #infer(fashion, person, virtual_fullpath)
	virtual_fullpath = fashion
        print("Did not find!")
    imgfile = {"fashion": fashion, "person": person, "virtual":  virtual_fullpath}
    feature_vector_main = encoder_predict(aimg)
    scores = [similarity_function(feature_vector_main, partner) for partner in allFeatures]
    closest = np.argsort(np.array(scores))
    topn=[]
    scoresCurrent = scores[closest[0]]
    for item in closest[5:2000]:
        #eliminate items that are too close of a match
       if scores[item] - scoresCurrent > 1:
           topn.append(item)
           scoresCurrent = scores[item]
    match = {}
    for idx, x in enumerate(topn[:8]):
        keyname = deepKeys[x]
        # keytype = keyname.split(os.sep)[1]
        match.update({"location{:02d}".format(idx + 1): os.path.join(z_img_dir, keyname)})
    return render_template("mirror_display.html", title="Smart Mirror App", match=match, imgfile=imgfile)


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'img_file_path' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['img_file_path']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgfile = {"person": "images/model_landing.jpg", "fashion": os.path.join("images", filename)}
            return render_template("index.html", title="Home", imgfile=imgfile)

    return


# Run inference
def infer(input_person, input_clothes, virtual_fit_output):
    """
    runs the inference virtual try on script for a person and clothing jpg image
    :param input_person: full path of input_person
    :param input_clothes: full path of input_clothes
    :param virtual_fit_output: desired full path of output
    :return:
    """
    run_smartfit = "./home/ubuntu/virtual_try-on/2d-Virtual-tryon-18B-AI.SV/run_smartfit.sh"
    fitdirectory = "/home/ubuntu/virtual_try-on/2d-Virtual-tryon-18B-AI.SV"
    os.system(" ".join([run_smartfit, input_person, input_clothes]))
    # Check that files exists (i.e. smartfit didn't crash)
    output_dir = os.sep.join([fitdirectory, 'output'])
    copyfile(os.sep.join([output_dir, 'output.png']), virtual_fit_output)
    return

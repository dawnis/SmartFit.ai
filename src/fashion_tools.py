import numpy as np
import cv2
from skimage.feature import hog
from joblib import Parallel, delayed
import pandas as pd

def fashion_similarity(input_txt, features, keys):
    """
    Computes the similarity metric between input and all features and returns the keys that are
    the most similar along with their score
    :param input_txt: txt name of image location
    :param features: feature vector
    :param keys: list of the keys of deepDict
    :return:
    """
    feature_index = keys.index(input_txt)
    input_vector = features[feature_index]

    scores = [similarity_function(input_vector, partner) for partner in features]
    return scores


def similarity_function(feature1, feature2):
    """
    Computes similarity using Manhattan distance. See code below for weighting some features over others (e.g. color)
    :param feature1: feature 1
    :param feature2: feature 2
    :return: similarity score
    """
    #256 HOG, 18 HSV, 512 Encoder
    salient1 = feature1[256:256+18].copy() #be careful not to modify feature vector in place
    salient2 = feature2[256:256+18].copy()
    feature1 =feature1.copy()
    feature2 = feature2.copy()
    #feature1[:] = 0
    #feature2[:] = 0
    feature1[256:256+18] = salient1*10
    feature2[256:256+18] = salient2*10
    abs_distance = np.abs(feature1 - feature2)
    return np.sum(abs_distance)


def similarity_function_old(feature1, feature2):
    """computes the similarity between two vectors"""
    f1Magnitude = feature1.dot(feature1)
    f2Magnitude = feature2.dot(feature2)
    return 1 - feature1.dot(feature2) / (f1Magnitude * f2Magnitude)


def rgb_image_bounding_box(image_full_path, boundingBox, convert_bgr=False):
    """
    Returns the rgb image cropped by bounding box
    :param image_full_path:
    :param bounding_box:
    :return:
    """
    imgraw = cv2.imread(image_full_path, 1)
    imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
    if convert_bgr:
        imgcrop = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2RGB)
    return imgcrop

# def image_to_feature_ae(image_full_path, boundingBox, encoder):
#     """version of image to feature that only has encoder in it"""
#     imgraw = cv2.imread(image_full_path, 1)
#     # no need to convert, keras autoencoder is in BGR color mode
#     imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
#     imgresize = cv2.resize(imgcrop, (128, 128))
#     imgresize = imgresize / 255.
#     imgresize = imgresize.astype('float32')
#     encoded_image = encoder.predict(imgresize[None, :, :, :])
#     encoded_image = encoded_image / np.max(encoded_image)
#     # print(fd)
#     # print(encoded_image)
#     return encoded_image.ravel()
#
#
# def image_to_feature_hog(image_full_path, boundingBox, encoder):
#     """version of image to feature that only has hog in it"""
#     imgraw = cv2.imread(image_full_path, 1)
#     # no need to convert, keras autoencoder is in BGR color mode
#     imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
#     imgresize = cv2.resize(imgcrop, (128, 128))
#     imgresize = imgresize / 255.
#     imgresize = imgresize.astype('float32')
#     grayscale = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
#     fd = hog(grayscale, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
#     fd = fd / np.max(fd)
#     return fd


def image_to_feature(image_full_path, boundingBox, encoder, features_to_use):
    """
    Converts cropped image of clothing to a feature vector
    :param image_full_path: Full path to fashion image
    :param boundingBox: bounding box of clothing within image, empty if no bounding box present
    :param encoder: trained autencoder model
    :param features_to_use: either "All" for everything or, one of: "encoder", "hog", "hsv"
    :return: a feature vector of length depending on features_to_use
    """
    imgraw = cv2.imread(image_full_path, 1)
    # no need to convert, keras autoencoder is in BGR color mode
    if len(boundingBox) > 0:
        imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
    else:
        imgcrop = imgraw
    imgresize = cv2.resize(imgcrop, (128, 128))
    imgresize = imgresize / 255.
    imgresize = imgresize.astype('float32')
    encoded_image = encoder.predict(imgresize[None, :, :, :])
    # print(input_img.shape)
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
    if features_to_use == "All":
        fv = np.concatenate((fd, hsv, encoded_image.ravel()))
    elif features_to_use == "encoder":
        fv = encoded_image.ravel()
    elif features_to_use == "hog":
        fv = fd
    elif features_to_use == "hsv":
        fv=hsv
    return fv


def generate_features(boxDict, img_directory, encoder):
    """
    generates feature vectors for all images in boxDict
    :param boxDict:
    :param encoder:
    :param scale:
    :return: feature vector
    """
    with Parallel(n_jobs=-1, verbose=2, backend="threading") as parallel:
        feature_vectors = parallel(delayed(image_to_feature)("/".join([img_directory, imagepath]), boundingBox, encoder)
                                   for imagepath, boundingBox in boxDict.items())
    return np.array(feature_vectors)


def cloth_category(cloth_txt):
    """
    Creates dictionary with name and numeric representation of which clothing category (e.g. tops, shorts, etc.)
    :param cloth_txt:
    :return:
    """
    category_cloth = {}
    linecount = 0
    with open(cloth_txt, 'r') as file:
        for linetext in file:
            line = linetext.rstrip(' \n')
            if linecount > 1:
                line_attributes = line.split("  ")
                category_cloth.update({line_attributes[0]: (linecount - 2, int(line_attributes[-1]))})
            linecount += 1
    return category_cloth


def category_cloth_img(cloth_img_txt):
    """
     returns category of each clothing image
    :param cloth_img_txt:
    :return:
    """
    categoryDict = {}
    linecount = 0
    with open(cloth_img_txt, 'r') as file:
        for linetext in file:
            # linetext = file.readline()
            line = linetext.rstrip(' \n')
            if linecount > 1:
                line_attributes = line.split(" ")
                categoryDict.update({line_attributes[0]: int(line_attributes[-1])})
            linecount += 1
    return categoryDict

def DeepFashion_Attributes(ClothCategory, img_attr_df):
    """
    Returns the attribute vectors associated with clothes in a particular category
    :param ClothCategory: the name of the cateogry
    :param img_attr_df: a pandas data frame (loaded from sparse save) of all attribute vectors
    :return:
    """
    cloth_category_txt = "labels/list_category_cloth.txt"
    cloth_img = "labels/list_category_img.txt"
    clothDict = cloth_category(cloth_category_txt)
    imgDict = category_cloth_img(cloth_img)
    #its putting in the index, not value!
    imgDF = pd.DataFrame.from_dict(imgDict, orient="index")
    imgDF.columns = ["ClothType"]
    img_attr_df = pd.concat([img_attr_df, imgDF], axis=1)
    clothIdx = clothDict[ClothCategory][0] + 1
    img_attr_df.drop(index = img_attr_df.index[img_attr_df["ClothType"] != clothIdx], inplace=True)
    img_attr_df.drop(columns = "ClothType", inplace=True)
    return img_attr_df

def DeepFashion(clothing_to_retrieve):
    """
    Returns dictionary where key is the image file name and the entry is the bounding box
    :param clothing_to_retrieve:
    :return: list and bounding boxes
    """
    cloth_category_txt = "labels/list_category_cloth.txt"
    cloth_img = "labels/list_category_img.txt"
    clothDict = cloth_category(cloth_category_txt)
    imgDict = category_cloth_img(cloth_img)
    category_index = clothDict[clothing_to_retrieve][0] + 1
    bounding_box = {}  # dictionary to return the bounding boxes of each piece of clothing
    linecount = 0
    with open('labels/list_bbox.txt', 'r') as file:
        for linetext in file:
            # linetext = file.readline()
            line = linetext.rstrip(' \n')
            if linecount > 1:
                line_attributes = line.split(" ")
                bounding_box.update({line_attributes[0]: [int(line_attributes[j]) for j in range(-4, 0)]})
            linecount += 1
    return {key: box for key, box in bounding_box.items() if imgDict[key] == category_index}

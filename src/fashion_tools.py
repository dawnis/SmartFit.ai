import numpy as np
import cv2
from skimage.feature import hog
from joblib import Parallel, delayed


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
    """computes the similarity between two vectors"""
    abs_distance = np.abs(feature1 - feature2)
    return np.sum(abs_distance)


def similarity_function_old(feature1, feature2):
    """computes the similarity between two vectors"""
    f1Magnitude = feature1.dot(feature1)
    f2Magnitude = feature2.dot(feature2)
    return 1 - feature1.dot(feature2) / (f1Magnitude * f2Magnitude)


def rgb_image_bounding_box(image_full_path, boundingBox):
    """
    Returns the rgb image cropped by bounding box
    :param image_full_path:
    :param bounding_box:
    :return:
    """
    imgraw = cv2.imread(image_full_path, 1)
    imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
    imgcrop = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2RGB)
    return imgcrop


def image_to_feature_ae(image_full_path, boundingBox, encoder):
    """version of image to feature that only has encoder in it"""
    imgraw = cv2.imread(image_full_path, 1)
    # no need to convert, keras autoencoder is in BGR color mode
    imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
    imgresize = cv2.resize(imgcrop, (128, 128))
    imgresize = imgresize / 255.
    imgresize = imgresize.astype('float32')
    encoded_image = encoder.predict(imgresize[None, :, :, :])
    encoded_image = encoded_image / np.max(encoded_image)
    # print(fd)
    # print(encoded_image)
    return encoded_image.ravel()


def image_to_feature_hog(image_full_path, boundingBox, encoder):
    """version of image to feature that only has hog in it"""
    imgraw = cv2.imread(image_full_path, 1)
    # no need to convert, keras autoencoder is in BGR color mode
    imgcrop = imgraw[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2], :]
    imgresize = cv2.resize(imgcrop, (128, 128))
    imgresize = imgresize / 255.
    imgresize = imgresize.astype('float32')
    grayscale = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
    fd = hog(grayscale, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
    fd = fd / np.max(fd)
    return fd


def image_to_feature(image_full_path, boundingBox, encoder):
    """from dict containing bounding box and names, returns feature vector"""
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
    # print(fd)
    # print(encoded_image)
    return np.concatenate((fd, encoded_image.ravel()))


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
    Creates dictionary with name and numeric representation
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

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import cv2
import glob
import copy
from skimage.feature import hog
from pathlib import Path
import random
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import librosa

from sklearn.decomposition import PCA


def get_hog(img_array = None):
  gray_img = cv2.cvtColor(((img_array)*255),cv2.COLOR_RGB2GRAY)
  hog_features, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualize=True)
  return hog_features, hog_image

def get_sift(img_array = None, num_features = 10):
  sift_final_des = np.zeros((num_features,128), dtype = np.float64)
  gray_img = cv2.cvtColor(((img_array)*255),cv2.COLOR_RGB2GRAY)
  sift = cv2.SIFT_create()
  sift_kp, sift_des = sift.detectAndCompute(gray_img.astype('uint8'),None)
  if sift_des is None:
    sift_des = np.zeros((num_features,128), dtype = np.float64)
  sift_des = np.nan_to_num(sift_des, nan=0, posinf=0, neginf=0)
  max_des = np.max(sift_des)
  max_des = 1
  if max_des is None or max_des == 0:
    max_des = 1
  sift_des_norm = sift_des/max_des
  sift_final_des[0:min(sift_des_norm.shape[0], num_features) , :] = sift_des_norm[0:min(sift_des_norm.shape[0], num_features) , :]
  return sift_final_des


def get_all_features(img, aud):
    img_file = cv2.imread(img)
    hog_features, hog_image = get_hog(img_array = img_file)
    #print(len(hog_features))
    sift_features = get_sift(img_array = img_file, num_features = 10)
    is0,is1 = sift_features.shape
    sift_features = sift_features.reshape(is0*is1)
    #print(sift_features.shape)
    
    y, sr = librosa.load(aud)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    as0,as1 = mfccs.shape
    mfccs = mfccs.reshape(as0*as1)
    #print(mfccs.shape)

    if len(hog_features) > 40000:
        hog_features = hog_features[0:40000]
    else:
        pad_length = 40000 - len(hog_features)
        hog_features = np.pad(hog_features, (0, pad_length), 'constant')

    if len(mfccs) > 300:
        mfccs = mfccs[0:300]
    else:
        pad_length = 300 - len(mfccs)
        mfccs = np.pad(mfccs, (0, pad_length), 'constant')
    
    features = np.concatenate((hog_features, sift_features,mfccs), axis=None).reshape(1,-1)
    #print(features_all.shape)
    #pca = PCA(n_components=n_components)
    #features = pca.fit_transform(features_all.reshape(1,-1))
    return features

def model_train(root_path):
    #dir_all_list = glob.glob('/home/ubuntu/food-101-small/Train/*')
    dir_all_list = glob.glob(root_path + '/*')
    # Ref - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#
    clf = SGDClassifier(loss="modified_huber")
    for ditem in dir_all_list:
        dir_lim_list = dir_all_list[:]
        dir_lim_list.remove(ditem)
        cur_path = ditem + '/*.jpg'
        print(ditem, flush=True)
        for fitem in glob.glob(cur_path):
            ritem = random.choice(dir_lim_list)
            cur_aud_path = ditem + '/' + str.split(ditem, '/')[-1] + '.wav'
            ran_aud_path = ritem + '/' + str.split(ritem, '/')[-1] + '.wav'
            X_pos = get_all_features(fitem, cur_aud_path)
            y_pos = np.array([[1]])
            clf.partial_fit(X_pos, y_pos.ravel(), classes=np.unique([[0], [1]]))
            X_neg = get_all_features(fitem, ran_aud_path)
            y_neg = np.array([[0]])
            clf.partial_fit(X_neg, y_neg.ravel(), classes=np.unique([[0], [1]]))
    print("\n")
    print("Model Training Completed. Saving the model.....", flush=True)

    with open('model.pkl','wb') as f:
        pickle.dump(clf,f)

def load_model(path):
    with open(path, 'rb') as f:
        clf = pickle.load(f)
        return clf
    

if __name__ == '__main__':
    train_path = '/home/ubuntu/food-101-small/Train'
    model_train(train_path)
    


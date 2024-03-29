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

def load_model(path):
    with open(path, 'rb') as f:
        clf = pickle.load(f)
        return clf

def eval_model(model_path, probe_path, gallery_path):
    aud_list = []
    img_list = []
    pred_list = []
    actual_list = []
    clf = load_model(model_path)
    print("Model Loaded Successfully", flush=True)
    probe_all_list = glob.glob(probe_path + '/*')
    gallery_all_list = glob.glob(gallery_path + '/*')
    count = 0
    for pitem in probe_all_list:
        cur_aud_path = pitem + '/' + str.split(pitem, '/')[-1] + '.wav'
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for gitem in gallery_all_list:
            for img in glob.glob(gitem + '/*.jpg'):
                '''
                #print(img, flush=True)
                if img not in img_list and cur_aud_path not in aud_list:
                    X = get_all_features(img, cur_aud_path)
                    y_pred = clf.predict(X)
                    #print(y_pred, flush=True)
                    if y_pred[0] == 1:
                        pred_list.append(y_pred)
                        img_list.append(img)
                        aud_list.append(cur_aud_path)
                        # break
                    label1 = str.split(pitem, '/')[-1]
                    label2 = str.split(img, '/')[-2]
                    if label1 == label2:
                        actual_list.append(np.array([1]))
                    else:
                        actual_list.append(np.array([0]))
                '''
                X = get_all_features(img, cur_aud_path)
                #y_pred = clf.predict(X)
                y_pred = clf.predict_proba(X)
                #print(y_pred[0][1])
                label1 = str.split(pitem, '/')[-1]
                label2 = str.split(img, '/')[-2]
                if label1 == label2:
                    tmp3.append(1.0)
                else:
                    tmp3.append(0.0)
                tmp1.append(img)
                tmp2.append(y_pred[0][1])
                break
        img_list.append(tmp1)
        actual_list.append(tmp3)
        pred_list.append(tmp2)
        aud_list.append(cur_aud_path)
        count = count + 1
        if count == 1:
            pass
            #break
    return pred_list, aud_list, img_list, actual_list


def get_cmc_scores(sim_matrix):
    #print(sim_matrix.shape)
    row_size, col_size = sim_matrix.shape
    rank_list = []
    cmc_scores = []
    for row in range(row_size):
      val = sim_matrix[row, :][row]
      rank = np.where(sorted(sim_matrix[row, :], reverse = True) == val)
      rank_list.append(rank[0][0]+1)
    current_score = 0
    for rs in range(row_size):
      current_score = current_score + (np.array(rank_list) == rs+1).sum()
      cmc_scores.append(current_score/row_size)
    return cmc_scores


def plot_cmc_curve(cmc_scores, path, figsize = (8,8)):
    x = np.arange(1,32)
    y = np.array(cmc_scores[0:31])
    plt.figure(figsize=figsize)
    plt.title("CMC Curve")
    plt.xlabel("Rank (m)")
    plt.ylabel("Rank-m Identification Accuracy")
    plt.xlim(1,10)
    plt.xticks(np.arange(1, 11, 1))
    plt.ylim(0,100)
    plt.yticks(np.arange(0, 100, 10))
    plt.plot(x, y*100, color ="green", linestyle='--', marker='o', label='line with marker')
    for x,y in zip(x,y):
        plt.annotate(str(round(y*100, 2)), xy=(x,y*100), xytext=(-7,7), textcoords='offset points')
    plt.grid(True)
    plt.legend(loc=2)
    plt.savefig(path, format = 'pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    model_path = '/home/ubuntu/model.pkl'
    probe_path = '/home/ubuntu/food-101-small/Probe'
    gallery_path = '/home/ubuntu/food-101-small/Gallery'
    pred_list, aud_list, img_list, actual_list = eval_model(model_path, probe_path, gallery_path)
    print("List of Predictions are :", pred_list, flush=True)
    print("List of Actual Values are :", actual_list, flush=True)
    print("\n", flush=True)
    print("Audio and Image Pairs are :", flush=True)
    print("\n", flush=True)
    print("Audio List :", aud_list, flush=True)
    print("\n", flush=True)
    print("Image List :", img_list, flush=True)
    cmc_scores = get_cmc_scores(np.array(pred_list))
    print("Rank1 Identification Accuracy:", cmc_scores[0]*100)
    plot_cmc_curve(cmc_scores, '/home/ubuntu/cmc_curve.pdf', (10,10))





    



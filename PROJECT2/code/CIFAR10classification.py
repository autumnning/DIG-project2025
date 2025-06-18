import numpy as np
import cv2
import skimage.feature
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

svm = SVC(kernel='linear', C=1.0, random_state=1)
pca = PCA(n_components=200)
cls_lr = LogisticRegression(random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)

def extract_feature(img):
    # return skimage.feature.local_binary_pattern(img, 8, 1.0, 'default') # LBP

    return skimage.feature.hog(img, 9, pixels_per_cell=[8, 8], cells_per_block=[2, 2], feature_vector=True) # HoG

    # _, res = skimage.filters.gabor(img, frequency=0.6) # Gabor
    # return res

def train():
    dir_path = './dataset'
    train_data = []
    labels = []
    with open(dir_path+'/train_labels.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            img = cv2.imread(dir_path + '/train/' + line[0], cv2.IMREAD_GRAYSCALE)

            # for LBP and Gabor filters
            # feature = extract_feature(img).reshape(1024,)

            # for HOG
            feature = extract_feature(img)

            train_data.append(feature)
            labels.append(line[-1])  # list of array

    # pca for feature reduction
    pca.fit(train_data)
    pca.transform(train_data)

    # svm, lr or knn
    # svm.fit(train_data, labels)
    cls_lr.fit(train_data, labels)
    # knn.fit(train_data, labels)

def test():
    dir_path = './dataset'
    test_data = []
    acc = 0
    cnt = 0
    with open(dir_path+'/test_labels.txt', 'r') as f:
        for line in f.readlines():
            cnt = cnt + 1
            line = line.strip('\n')
            line = line.split()
            test_data.append(line)
            img = cv2.imread(dir_path + '/test/' + line[0], cv2.IMREAD_GRAYSCALE)

            # for LBP and Gabor filters
            # feature = extract_feature(img).reshape(-1, 1024)

            # for HOG
            feature = extract_feature(img)
            feature = feature.reshape(-1, len(feature))

            pca.transform(feature)

            # svm, lr or knn, must same as train
            # pre = svm.predict(feature)
            pre = cls_lr.predict(feature)
            # pre = knn.predict(feature)

            if pre == line[-1]:
                acc = acc+1
    print(f'test accuracy rate: %.1f%%'%(acc/cnt*100))


if __name__ == '__main__':

    train()

    test()

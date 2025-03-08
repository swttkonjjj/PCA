import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import cv2
import random
from matplotlib.font_manager import FontProperties
from numpy.core.fromnumeric import mean  # 字体管理器

# 设置汉字格式
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)


def loadImage(image_path):
    # print(image_path)
    image = cv2.imread(image_path, -1)
    # print(image.shape)
    # cv2.imshow(image)
    image = image.flatten()
    # print(image.shape)
    return image


def loadData(test_image_path, data_path="orl_faces"):
    train_data = []
    train_lable = []
    test_data = []
    test_data.append(loadImage(test_image_path))
    # for train_item in train:
    for i in range(1, 41):
        train_path = data_path + "\\s" + str(i)
        for j in range(1, 11):
            train_image_path = train_path + "\\" + str(j) + ".pgm"
            if test_image_path == train_image_path:
                continue
            # print(train_image_path)
            train_data.append(loadImage(train_image_path))
            train_lable.append(train_image_path)
    return train_data, train_lable, test_data


def pca(date_mat, max_rank=200):
    date_mat = np.float32(np.mat(date_mat))
    mean_value = np.mean(date_mat, axis=0)
    mean_removed = date_mat - mean_value
    # cov_mat = mean_removed * mean_removed.T
    cov_mat = np.cov(mean_removed, rowvar=1)
    eig_vals, eig_vects = np.linalg.eig(cov_mat)
    sort_vals = np.argsort(eig_vals)
    select_vals = sort_vals[: -(max_rank + 1): -1]
    select_vects = eig_vects[:, select_vals]
    select_vects = mean_removed.T * select_vects
    lowD = mean_removed * select_vects
    return lowD, select_vects, mean_value


# def pca(dataMat, topNfeat=100):
#     meanVals = np.mean(dataMat, axis=0)
#     meanRemoved = dataMat - meanVals
#     covMat = np.cov(meanRemoved, rowvar=0)
#     eigVals, eigVects = linalg.eig(np.mat(covMat))
#     eigVlInd = np.argsort(eigVals)
#     eigVlInd = eigVlInd[:-(topNfeat+1) : -1]
#     redEigVects = eigVects[:,eigVlInd]
#     lowDDataMat = meanRemoved * redEigVects
#     # reconmat = (lowDDataMat * redEigVects.T) + meanVals
#     return lowDDataMat, redEigVects


def knn(inX, dataSet, labels, k):
    inX = np.array(inX)
    dataSet = np.array(dataSet)
    labels = np.array(labels)

    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sorteedDisttTndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sorteedDisttTndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=lambda d: d[1], reverse=True)
    return sortedClassCount[0][0]

def main():
    person = random.randint(1, 40)
    picture = random.randint(1, 10)
    test_face = "orl_faces\\s" + str(person) + "\\" + str(picture) + ".pgm"

    train_data, train_lable, test_data = loadData(test_face)
    lowD, select_vects, mean_value = pca(train_data)
    test_data -= mean_value
    test_data = np.mat(test_data) * np.mat(select_vects)
    best_match = knn(test_data, lowD, train_lable, 1)

    # image = cv2.imread("orl_faces\\s1\\1.pgm", -1)
    image = cv2.imread(test_face, -1)
    image2 = cv2.imread(best_match, -1)
    # cv2.imshow(image)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("随机测试人脸", fontproperties=font)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap="gray")
    plt.title("最相似人脸", fontproperties=font)
    # 隐藏坐标系
    plt.axis('off')
    # 展示图片
    plt.show()
    # image = cv2.imread('orl_faces/s1/1.pgm', -1)
    # print(image)

if __name__ =="__main__":
    main() 
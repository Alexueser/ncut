# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize


def similarity_function(points):
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def normalized_cut(points, k):
    A = similarity_function(points)
    W = np.eye(len(A)) - normalize(A, norm='l1')
    eigvalues, eigvectors = LA.eig(W)
    indices = np.argsort(eigvalues)[1:k]
    return KMeans(n_clusters=k).fit_predict(eigvectors[:, indices])


X, y = make_blobs()
labels = normalized_cut(X, 3)
# 画图
plt.style.use('ggplot')
# 原数据
fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.scatter(X[:, 0], X[:, 1], c=y)
ax0.set_title('raw data')
# 谱聚类结果
ax1.scatter(X[:, 0], X[:, 1], c=labels)
ax1.set_title('Normalized Cut')
plt.show()
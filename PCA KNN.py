#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score,accuracy_score


# In[19]:


# 在stdout上显示进度日志
print(__doc__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# In[4]:


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# In[5]:


# 对图像数组进行训练以找到用于绘制的形状
n_samples, h, w = lfw_people.images.shape


# In[6]:


#对于机器学习，我们直接使用2个数据（作为相对像素
#这个模型忽略了位置信息
X = lfw_people.data
n_features = X.shape[1]


# In[7]:


#要预测的标签是该人的id
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


# In[8]:


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# In[9]:


#使用分层k折法将训练集和测试集分开

#分成训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[10]:


#将其视为未标记，在人脸数据集上计算PCA（特征脸）

#数据集）：无监督特征提取/降维
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))


# In[11]:


t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))


# In[12]:


eigenfaces = pca.components_.reshape((n_components, h, w))


# In[13]:


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# In[14]:


#建立KNN分类模型
# KNN.fit(X_train_pca, y_train)

param_grid = {'n_neighbors':[1,3,5,10] }

model = GridSearchCV(
    KNeighborsClassifier(), param_grid
)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
print("Best estimator found by grid search:")
print(model.best_estimator_)


# In[15]:


cnf_matrix =confusion_matrix(y_test,y_pred)
#输出模型精度，这个可没有逻辑回归的recall

print(accuracy_score(y_test,y_pred))
cnf_matrix


# In[16]:


print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# In[17]:


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# In[18]:


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


# In[19]:


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





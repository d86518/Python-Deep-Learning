
# coding: utf-8
import os
os.chdir('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes')

import keras
keras.__version__ # 2.0.8 -> 2.1.2


# # 卷積神經網路簡介
# 
# - 以簡單卷積神經網路(convnet)分類MNIST手寫數字
# - 輸入灰階影像(28,28,1)(`(image_height, image_width, image_channels)`)，也就是說第一層的`input_shape = c(28, 28, 1)`
# - 三個二維卷積層(`layer_conv_2d()`)穿插兩個最大池化層(`layer_max_pooling_2d()`)


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# - model.summary()報導當前convnet網路架構


model.summary()


# - 各層(Conv2D與MaxPooling2D)輸出為3D張量(height, width, channels)，height和width會漸次縮小當網路逐漸走深，channels是layers.Conv2D()函數傳入的第一個參數(e.g. 32 or 64).
# 
# - 下一個步驟首先扁平化3D輸出(3, 3, 64)為1D，然後再傳入稠密層或完全聯通層(Dense layers)，最後是十類輸出值並經softmax轉換為各類別機率值進行預測.


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# - 網路最終架構如下：


model.summary()


# - 如前所述，(3, 3, 64)3D張量在通過兩層稠密層之前，先壓平為1D的(576,)張量
# - 定義完成後我們載入MNIST進行訓練


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

import numpy as np
print(train_labels.shape)
np.unique(train_labels) # 0~9

train_labels = to_categorical(train_labels)
train_labels.shape

test_labels = to_categorical(test_labels)



model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)


dir()


history.history # a dict about accuracy and loss


history_dict = history.history
history_dict.keys()


# - 以測試資料評估模型

test_loss, test_acc = model.evaluate(test_images, test_labels)


test_loss


test_acc


# - 比先前稠密連通的神經網路表現更好！

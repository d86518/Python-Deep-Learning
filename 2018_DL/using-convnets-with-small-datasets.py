
# coding: utf-8
import os
os.chdir('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes')

import keras
keras.__version__ # '2.0.8'


# # 遷移學習 Transfer Learning
# 
# 遷移學習(Transfer learning) 顧名思義就是就是把已學訓練好的模型參數遷移到新的模型來幫助新模型訓練。考慮到大部分數據或任務是存在相關性的，所以通過遷移學習我們可以將已經學到的模型參數（也可理解為模型學到的知識）通過某種方式來分享給新模型從而加快並最佳化模型的學習效率，不用像大多數類神經網路那樣從零學習（starting from scratch，tabula rasa）。(作者：劉詩昆，鏈接：https://www.zhihu.com/question/41979241/answer/123545914
# 來源：知乎)

# ## convnets的小資料集建模
# 
# - 深度學習亦可用於非大量的資料(數百或數萬張圖片)
# - Kaggle Cats vs. Dogs資料集有4000張貓狗圖片，數量是貓狗各半
# - 我們將用2000張圖片先訓練無係數正規化(regularization)的基本模型(baseline)，1000張為核驗集，1000張測試集 -> 正確率為71%
# - 為避免過度配適，將介紹電腦視覺中減緩過度配適的data augmentation -> 正確率為82%

# - 另外，應用深度學習到非大量資料的兩項必要技術是：以預先訓練好的神經網路(*a pre-trained network*)進行屬性提取(90% -> 93%)，以及微調(*fine-tuning*)預先訓練好的神經網路(93% -> 95%)

# ## 深度學習不只適用於大量資料(應用遷移學習)
# 
# - 大量資料的情況下，深度學習可從訓練資料中找到有趣的屬性，無需人工進行屬性工程。但是這只有在訓練樣本充足，且樣本為高維度數據(例如圖片)時方為如此
# - 深度學習模型本質上是多用途的，依大規模資料集訓練完成後的圖片分類或語音轉文字等模型，可以在截然不同的問題上做小幅修改後重複使用
# - 尤其在電腦視覺領域中，許多預先訓練好的模型(通常是以ImageNet資料集訓練的)，都可公開下載取用

# ## 資料下載
# 
# - 2013年Kaggle.com的電腦視覺競賽中貓狗資料集(`https://www.kaggle.com/c/dogs-vs-cats/data`)(須先建立Kaggle帳戶).
# - 這些圖片是中解析度的彩色JPEG檔:
# 
# ![cats_vs_dogs_samples](https://s3.amazonaws.com/book.keras.io/img/ch5/cats_vs_dogs_samples.jpg)

# - 原始資料集有25,000張貓狗圖片(各12,500張)，壓縮後543MB，請存放在'./kaggle_original_data'
# - 下載後解壓縮之，並分成三個資料夾存放：各類1000張的訓練集、各類500張的核驗集、以及各類500張的測試集


import os, shutil


# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = '/Users/Vince/Downloads/kaggle_original_data'

# The directory where we will
# store our smaller dataset
base_dir = '/Users/Vince/Downloads/cats_and_dogs_small'
os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


# - i從1到1000填入大括弧中

help(shutil.copyfile)


original_dataset_dir = '/Users/Vince/Downloads/kaggle_original_data'
# The directory where we will
# store our smaller dataset
base_dir = '/Users/Vince/Downloads/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')


# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname) # 來源資料夾與檔案
    dst = os.path.join(train_cats_dir, fname) # 目的資料夾與檔案
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# - 檢查各個資料夾中有多少張圖片(train/validation/test):

print('total training cat images:', len(os.listdir(train_cats_dir)))


print('total training dog images:', len(os.listdir(train_dogs_dir)))


print('total validation cat images:', len(os.listdir(validation_cats_dir)))


print('total validation dog images:', len(os.listdir(validation_dogs_dir)))


print('total test cat images:', len(os.listdir(test_cats_dir)))


print('total test dog images:', len(os.listdir(test_dogs_dir)))


# - 三個集合的兩類樣本數均相同，因此這是一個平衡的二元分類問題，所以分類正確率是一個合適的模型評量指標

# ## 建立convnets
# 
# - 同前的一般性架構，些許差異的`Conv2D`(具`relu`活化函數)與`MaxPooling2D`層堆疊而成的神經網路.
# - 然而因為影像更大與問題更複雜，將網路調整為*多一層*的`Conv2D`+`MaxPooling2D`層，使其*到達`Flatten`層時不會過大*.
# - 從150x150的輸入大小到`Flatten`層前的7x7.
# - 屬性圖的深度從32到128逐漸增加，而其大小從148x148到7x7逐漸減少，幾乎所有的convnets都是如此.
# - 因為問題是二元分類問題，所以網路最後是具`sigmoid`活化函數的單一節點稠密層，已取得某類的機率值.


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# - 以model.summary()函數檢視各層屬性圖其維度的變化.


model.summary()


# For our compilation step, we'll go with the `RMSprop` optimizer as usual. Since we ended our network with a single sigmoid unit, we will 
# use binary crossentropy as our loss (as a reminder, check out the table in Chapter 4, section 5 for a cheatsheet on what loss function to 
# use in various situations).


from keras import optimizers
dir(optimizers)

#- RMSprop最佳化算法適合遞歸式類神經網路。
#- lr: float >= 0. Learning rate.
help(optimizers.RMSprop) # This optimizer is usually a good choice for recurrent neural networks.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# ## 資料前處理 Data preprocessing
# 
# As you already know by now, data should be formatted into appropriately pre-processed floating point tensors before being fed into our 
# network. Currently, our data sits on a drive as JPEG files, so the steps for getting it into our network are roughly:
# 
# * 讀取圖片檔 Read the picture files.
# * Decode the JPEG content to RBG grids of pixels.
# * Convert these into floating point tensors.
# * Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).
# 
# It may seem a bit daunting, but thankfully Keras has utilities to take care of these steps automatically. Keras has a module with image 
# processing helper tools, located at `keras.preprocessing.image`. In particular, it contains the class `ImageDataGenerator` which allows to 
# quickly set up Python generators that can automatically turn image files on disk into batches of pre-processed tensors. This is what we 
# will use here.


from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


# Let's take a look at the output of one of these generators: it yields batches of 150x150 RGB images (shape `(20, 150, 150, 3)`) and binary 
# labels (shape `(20,)`). 20 is the number of samples in each batch (the batch size). Note that the generator yields these batches 
# indefinitely: it just loops endlessly over the images present in the target folder. For this reason, we need to `break` the iteration loop 
# at some point.


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# Let's fit our model to the data using the generator. We do it using the `fit_generator` method, the equivalent of `fit` for data generators 
# like ours. It expects as first argument a Python generator that will yield batches of inputs and targets indefinitely, like ours does. 
# Because the data is being generated endlessly, the generator needs to know example how many samples to draw from the generator before 
# declaring an epoch over. This is the role of the `steps_per_epoch` argument: after having drawn `steps_per_epoch` batches from the 
# generator, i.e. after having run for `steps_per_epoch` gradient descent steps, the fitting process will go to the next epoch. In our case, 
# batches are 20-sample large, so it will take 100 batches until we see our target of 2000 samples.
# 
# When using `fit_generator`, one may pass a `validation_data` argument, much like with the `fit` method. Importantly, this argument is 
# allowed to be a data generator itself, but it could be a tuple of Numpy arrays as well. If you pass a generator as `validation_data`, then 
# this generator is expected to yield batches of validation data endlessly, and thus you should also specify the `validation_steps` argument, 
# which tells the process how many batches to draw from the validation generator for evaluation.


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# It is good practice to always save your models after training:


model.save('./models/cats_and_dogs_small_0.h5') # _0: mine, _1: Elias's

model_1 = keras.models.load_model('./models/cats_and_dogs_small_0.h5')

# Let's plot the loss and accuracy of the model over the training and validation data during training:

import pickle
f = open("./models/cats_and_dogs_small_0.txt", "rb") # "rb": read in binary
history = pickle.load(f)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# These plots are characteristic of overfitting. Our training accuracy increases linearly over time, until it reaches nearly 100%, while our 
# validation accuracy stalls at 70-72%. Our validation loss reaches its minimum after only five epochs then stalls, while the training loss 
# keeps decreasing linearly until it reaches nearly 0.
# 
# Because we only have relatively few training samples (2000), overfitting is going to be our number one concern. You already know about a 
# number of techniques that can help mitigate overfitting, such as dropout and weight decay (L2 regularization). We are now going to 
# introduce a new one, specific to computer vision, and used almost universally when processing images with deep learning models: *data 
# augmentation*.

# ## Using data augmentation
# 
# Overfitting is caused by having too few samples to learn from, rendering us unable to train a model able to generalize to new data. 
# Given infinite data, our model would be exposed to every possible aspect of the data distribution at hand: we would never overfit. Data 
# augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number 
# of random transformations that yield believable-looking images. The goal is that at training time, our model would never see the exact same 
# picture twice. This helps the model get exposed to more aspects of the data and generalize better.
# 
# In Keras, this can be done by configuring a number of random transformations to be performed on the images read by our `ImageDataGenerator` 
# instance. Let's get started with an example:


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# These are just a few of the options available (for more, see the Keras documentation). Let's quickly go over what we just wrote:
# 
# * `rotation_range` is a value in degrees (0-180), a range within which to randomly rotate pictures.
# * `width_shift` and `height_shift` are ranges (as a fraction of total width or height) within which to randomly translate pictures 
# vertically or horizontally.
# * `shear_range` is for randomly applying shearing transformations.
# * `zoom_range` is for randomly zooming inside pictures.
# * `horizontal_flip` is for randomly flipping half of the images horizontally -- relevant when there are no assumptions of horizontal 
# asymmetry (e.g. real-world pictures).
# * `fill_mode` is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
# 
# Let's take a look at our augmented images:


# This is module with image preprocessing utilities
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()


# If we train a new network using this data augmentation configuration, our network will never see twice the same input. However, the inputs 
# that it sees are still heavily intercorrelated, since they come from a small number of original images -- we cannot produce new information, 
# we can only remix existing information. As such, this might not be quite enough to completely get rid of overfitting. To further fight 
# overfitting, we will also add a Dropout layer to our model, right before the densely-connected classifier:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) # 新增丟棄層
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# Let's train our network using data augmentation and dropout:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


# Let's save our model -- we will be using it in the section on convnet visualization.


model.save('cats_and_dogs_small_2.h5')

model_recovery = keras.models.load_model("./models/cats_and_dogs_small_2.h5")

import pickle
f = open("./models/cats_and_dogs_small_2.txt", "rb") # "rb": read in binary
history = pickle.load(f)


# Let's plot our results again:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Thanks to data augmentation and dropout, we are no longer overfitting: the training curves are rather closely tracking the validation 
# curves. We are now able to reach an accuracy of 82%, a 15% relative improvement over the non-regularized model.
# 
# By leveraging regularization techniques even further and by tuning the network's parameters (such as the number of filters per convolution 
# layer, or the number of layers in the network), we may be able to get an even better accuracy, likely up to 86-87%. However, it would prove 
# very difficult to go any higher just by training our own convnet from scratch, simply because we have so little data to work with. As a 
# next step to improve our accuracy on this problem, we will have to leverage a pre-trained model, which will be the focus of the next two 
# sections.

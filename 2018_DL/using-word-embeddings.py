
# coding: utf-8


import os
os.chdir('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes')

import keras
keras.__version__


# # 字詞嵌入 Using word embeddings
# 
# - 詞嵌入(word embeddings)是另一種將字詞關聯到向量(相對於單熱編碼，詞嵌入關聯到稠密向量)的另一種強大且受歡迎的方法，前述單熱編碼後的結果是二元、稀疏且高維的(維度為詞彙的字數)向；詞嵌入則是低維的浮點數向量，亦即其結果為稠密向量。另一個與單熱編碼不同之處是詞嵌入由資料中學習出各文本的屬性向量，當詞彙非常多時常用的詞嵌入維度為256、512或1024。而單熱編碼其維度常高於20,000，所以詞嵌入可理解為將更多的訊息封裝在非常少的維度裡。
# - 其實詞嵌入是做迴歸，其參數是中心詞w_{i}與鄰居詞w_{j}的向量，配適目標為最大化條件機率P(w_{j}|w_{i})，或是最大化二元字詞機率P(w_{i}, w_{j})。

# ![word embeddings vs. one hot encoding](https://s3.amazonaws.com/book.keras.io/img/ch6/word_embeddings.png)

# > 詞嵌入的兩種方式：

# - 隨機初始化字詞向量，在機器學習的過程中(例如：文件分類或情感分析)同步學習詞嵌入向量，如同類神經網路學習其權重一般。
# - 分開學習詞嵌入向量，這種方式被稱為預先訓練的詞嵌入(pre-trained word embeddings)。

# ## 以嵌入層學習詞嵌入

# - 將字詞關聯到稠密向量的最簡單方式是隨機選取向量元素值，但是此法導致嵌入空間無結構，例如：語意相近的accurate和exact兩字有完全不同的嵌入值。深度神經網路難以理解此一嘈雜且無結構的嵌入空間。
# - 比較好的方式是讓字詞(嵌入)向量間的幾何關係反映它們之間的語意關係，同義字相近而異義字較遠，且嵌入空間中的特定方向 __directions__ 是有意義的，並可進行運算。例如：女性一詞的詞嵌入向量加上國王的詞嵌入向量，就產生王后的詞嵌入向量，詞嵌入空間通常有數千個可解釋且有用的字詞向量。

# > 可能的問題
# 
# - 是否有理想的詞嵌入空間？
# - 不同的語言是非同構的(not isomorphic)!
# - 特定語意關係的重要度依不同任務而有所不同。
# - 新任務得學習新的嵌入空間。

# ## IMDB電影評論情感預測任務

# keras以layer_embedding初始化嵌入層時，其權重一開始是隨機的。訓練期間詞嵌入向量透過倒傳遞算法逐步調整，架構出後續模型可以利用的詞嵌入空間，也就是說其空間結構適合當前的問題。
# 
# - 限定最常見的10,000個字
# - 每篇評論僅看20個字詞
# - 詞嵌入空間維度為8，將輸入的2D整數值張量，轉為3D嵌入浮點數張量
# - 再將3D張量扁平化為2D，其上以單一稠密層完成分類任務



from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)


# - 詞嵌入層`Embedding`可以理解為將字詞整數值索引(代表特定字詞)，到一稠密值向量的映射關係。Python可以字典dict結構儲存此映射關係，給定字詞整數索引，字典回傳對應的向量，此即為Python的字典查詢(dictionary lookup)。

# - 詞嵌入層`Embedding`是從各文件與字詞索引(Word index)的二維向量`(samples, sequence_length)`，加進各字詞原長度不等`(sequence_length, variable_embedding_dimensionality)`，但經補綴與截略處理的嵌入向量後，成為三維的文件字詞嵌入空間張量`(samples, sequence_length, embedding_dimensionality)`。
# - 三維的文件字詞嵌入空間張量`(samples, sequence_length, embedding_dimensionality)`可以投入遞歸層RNN或一維卷積層1D_convnet。
# - Word index`(samples, sequence_length)` -> Embedding layer`(sequence_length, variable_embedding_dimensionality)` -> Padding and Truncating -> Corresponding word vector`(samples, sequence_length, embedding_dimensionality)` -> An RNN layer or a 1D convnet
# 
# - 如同其它類型的類神經網路層，詞嵌入層`Embedding`一開始會隨機初始化其權重，接著訓練過程中以倒傳遞(backpropagation)演算法逐步調整權重值，最終架構出詞嵌入空間。
# - 以下準備IMDB電影評論資料，以前10,000個字為詞彙，每篇評論在20個字後截略，每個字都使用8維的詞嵌入空間`(25,000, 20, 8)`。
# - 上面3D浮點數張量經`Flatten()`壓平為2D後，再以單一稠密層`Dense()`進行分類工作。



from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words 
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)




from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))
# After the Embedding layer, 
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings 
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# - 核驗集準確率約略為75%，別忘了每篇評論我們只看20個字。
# - 而且將嵌入層扁平化後，僅在其上疊加單一的一層稠密層，個別考慮字詞忽略字詞間關係與句法結構。
# - 更好的做法可能是加上遞歸層或1D卷積層以學習字詞順序的屬性。
# - 運用預先訓練好的詞嵌入資料庫word2vector或GloVe。

# ## Using pre-trained word embeddings
# 
# - word2vector by Thomas Mikilov at Google
# - GloVe (Global Vectors for Word Representation) by Standford univ.
# 
# Sometimes, you have so little training data available that could never use your data alone to learn an appropriate task-specific embedding 
# of your vocabulary. What to do then?
# 
# Instead of learning word embeddings jointly with the problem you want to solve, you could be loading embedding vectors from a pre-computed 
# embedding space known to be highly structured and to exhibit useful properties -- that captures generic aspects of language structure. The 
# rationale behind using pre-trained word embeddings in natural language processing is very much the same as for using pre-trained convnets 
# in image classification: we don't have enough data available to learn truly powerful features on our own, but we expect the features that 
# we need to be fairly generic, i.e. common visual features or semantic features. In this case it makes sense to reuse features learned on a 
# different problem.
# 
# Such word embeddings are generally computed using word occurrence statistics (observations about what words co-occur in sentences or 
# documents), using a variety of techniques, some involving neural networks, others not. The idea of a dense, low-dimensional embedding space 
# for words, computed in an unsupervised way, was initially explored by Bengio et al. in the early 2000s, but it only started really taking 
# off in research and industry applications after the release of one of the most famous and successful word embedding scheme: the Word2Vec 
# algorithm, developed by Mikolov at Google in 2013. Word2Vec dimensions capture specific semantic properties, e.g. gender.
# 
# There are various pre-computed databases of word embeddings that can download and start using in a Keras `Embedding` layer. Word2Vec is one 
# of them. Another popular one is called "GloVe", developed by Stanford researchers in 2014. It stands for "Global Vectors for Word 
# Representation", and it is an embedding technique based on factorizing a matrix of word co-occurrence statistics. Its developers have made 
# available pre-computed embeddings for millions of English tokens, obtained from Wikipedia data or from Common Crawl data.
# 
# Let's take a look at how you can get started using GloVe embeddings in a Keras model. The same method will of course be valid for Word2Vec 
# embeddings or any other word embedding database that you can download. We will also use this example to refresh the text tokenization 
# techniques we introduced a few paragraphs ago: we will start from raw text, and work our way up.

# ## Putting it all together: from raw text to word embeddings
# 
# 
# We will be using a model similar to the one we just went over -- embedding sentences in sequences of vectors, flattening them and training a 
# `Dense` layer on top. But we will do it using pre-trained word embeddings, and instead of using the pre-tokenized IMDB data packaged in 
# Keras, we will start from scratch, by downloading the original text data.

# ### Download the IMDB data as raw text
# 
# 
# First, head to `http://ai.stanford.edu/~amaas/data/sentiment/` and download the raw IMDB dataset (if the URL isn't working anymore, just 
# Google "IMDB dataset"). Uncompress it.
# 
# Now let's collect the individual training reviews into a list of strings, one string per review, and let's also collect the review labels 
# (positive / negative) into a `labels` list:



import os

imdb_dir = '/home/ubuntu/data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# ### Tokenize the data
# 
# 
# Let's vectorize the texts we collected, and prepare a training and validation split.
# We will merely be using the concepts we introduced earlier in this section.
# 
# Because pre-trained word embeddings are meant to be particularly useful on problems where little training data is available (otherwise, 
# task-specific embeddings are likely to outperform them), we will add the following twist: we restrict the training data to its first 200 
# samples. So we will be learning to classify movie reviews after looking at just 200 examples...
# 



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # We will cut reviews after 100 words
training_samples = 200  # We will be training on 200 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# ### Download the GloVe word embeddings
# 
# 
# Head to `https://nlp.stanford.edu/projects/glove/` (where you can learn more about the GloVe algorithm), and download the pre-computed 
# embeddings from 2014 English Wikipedia. It's a 822MB zip file named `glove.6B.zip`, containing 100-dimensional embedding vectors for 
# 400,000 words (or non-word tokens). Un-zip it.

# ### Pre-process the embeddings
# 
# 
# Let's parse the un-zipped file (it's a `txt` file) to build an index mapping words (as strings) to their vector representation (as number 
# vectors).



glove_dir = '/home/ubuntu/data/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# 
# Now let's build an embedding matrix that we will be able to load into an `Embedding` layer. It must be a matrix of shape `(max_words, 
# embedding_dim)`, where each entry `i` contains the `embedding_dim`-dimensional vector for the word of index `i` in our reference word index 
# (built during tokenization). Note that the index `0` is not supposed to stand for any word or token -- it's a placeholder.



embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# ### Define a model
# 
# We will be using the same model architecture as before:



from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# ### Load the GloVe embeddings in the model
# 
# 
# The `Embedding` layer has a single weight matrix: a 2D float matrix where each entry `i` is the word vector meant to be associated with 
# index `i`. Simple enough. Let's just load the GloVe matrix we prepared into our `Embedding` layer, the first layer in our model:



model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False # 固定嵌入層不進行學習！


# 
# Additionally, we freeze the embedding layer (we set its `trainable` attribute to `False`), following the same rationale as what you are 
# already familiar with in the context of pre-trained convnet features: when parts of a model are pre-trained (like our `Embedding` layer), 
# and parts are randomly initialized (like our classifier), the pre-trained parts should not be updated during training to avoid forgetting 
# what they already know. The large gradient update triggered by the randomly initialized layers would be very disruptive to the already 
# learned features.

# ### Train and evaluate
# 
# Let's compile our model and train it:



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')


# Let's plot its performance over time:



import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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


# 
# The model quickly starts overfitting, unsurprisingly given the small number of training samples. Validation accuracy has high variance for 
# the same reason, but seems to reach high 50s.
# 
# Note that your mileage may vary: since we have so few training samples, performance is heavily dependent on which exact 200 samples we 
# picked, and we picked them at random. If it worked really poorly for you, try picking a different random set of 200 samples, just for the 
# sake of the exercise (in real life you don't get to pick your training data).
# 
# We can also try to train the same model without loading the pre-trained word embeddings and without freezing the embedding layer. In that 
# case, we would be learning a task-specific embedding of our input tokens, which is generally more powerful than pre-trained word embeddings 
# when lots of data is available. However, in our case, we have only 200 training samples. Let's try it:



from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))




acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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


# 
# Validation accuracy stalls in the low 50s. So in our case, pre-trained word embeddings does outperform jointly learned embeddings. If you 
# increase the number of training samples, this will quickly stop being the case -- try it as an exercise.
# 
# Finally, let's evaluate the model on the test data. First, we will need to tokenize the test data:



test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)


# And let's load and evaluate the first model:



model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)


# We get an appalling test accuracy of 54%. Working with just a handful of training samples is hard!

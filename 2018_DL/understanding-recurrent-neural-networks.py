
# coding: utf-8


import os
os.chdir('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes')

import keras
keras.__version__


# # 瞭解遞歸神經網路

# - 類神經網路(稠密連結NN與卷積NN)大多沒有記憶！每個投入被獨立處理，未保留個投入間的狀態。
# - 獨立處理的狀況下，我們必須一次丟入一整筆序列的資料，作為單一的資料點。
# - 一筆完整的電影評論轉換為單一的大型數值向量，再丟入前向式類神經網路。
# - 請思考我們平常的閱讀習慣：您是否會來來回回地閱讀文章？
# - 漸增式的訊息處理：根據過去的訊息建立當前的情報與狀態，在新訊息不斷進來時更新情報，並在適當的時點重設狀態，例如：兩不同評論間重設狀態。
# - 遞歸神經網路(recurrent neural networks)：透過序列資料的元素進行迭代，依據截至目前的已見，維護狀態資訊，連同源源不絕的新訊息進行處理。

# ![NetworkWithALoop](_img/ANetworkWithALoop.png)

# - 遞歸神經網絡（RNN）是兩種人工神經網絡的總稱，一種是時間遞歸神經網絡（recurrent neural network），另一種是結構遞歸神經網絡（recursive neural network）。時間遞歸神經網絡的神經元間連接構成矩陣，而結構遞歸神經網絡利用相似的神經網絡結構遞歸構造更為複雜的深度網絡。RNN一般指代時間遞歸神經網絡。單純遞歸神經網絡因為無法處理隨著遞歸，權重指數級爆炸或消失的問題（Vanishing gradient problem），難以捕捉長期時間關聯；而結合不同長短期記憶的LSTM類神經網路(Long-Short Term Memory, LSTM)可以很好解決這個問題。(https://zh.wikipedia.org/wiki/%E9%80%92%E5%BD%92%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

# ![ASimpleRNN](_img/ASimpleRNN.png)

# ## Keras中的遞歸神經網路
# 
# - Keras的`SimpleRNN`層:



from keras.layers import SimpleRNN


# - keras中的`SimpleRNN`直接處理一批批的序列資料，所以其3D投入資料的各維長度為`(batch_size, timesteps, input_features)`而非`(timesteps, input_features)`。
# - Keras所有遞歸層(`SimpleRNN`亦不例外)均可以兩種模式執行：回傳各時序步驟下的完整輸出(3D張量`(batch_size, timesteps, output_features)`)；或者僅傳回最後時序下的輸出`(batch_size, output_features)`，此兩種模式可由`return_sequences`建構子來控制。



from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32)) # 32*32 + 32*32 + 32*1 = 2080
model.summary()



model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()


# - 有時會堆疊數個遞歸層，以增加網路的知識表達能力，此時return_sequences必須都設為True。



model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()


# - 接著嘗試以IMDB電影評論資料配適上面的RNN分類模型，首先先準備資料：



from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# - 詞嵌入層`Embedding`後是單一的遞歸層`SimpleRNN`：



from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


# - 將模型儲存至 HDF5 檔案中(https://blog.gtwang.org/programming/keras-save-and-load-model-tutorial/)



model.save("SimpleRNN.h5") # creates a HDF5 file 'my_model.h5'

import pickle
with open("./models/SimpleRNN.txt", "wb") as f:
    pickle.dump(history.history, f) # history會有TypeError:

# - 繪圖顯示訓練集與核驗集的損失函數值與正確率。



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




type(history)




dir(history)


# - 在模型儲存至 HDF5 檔案之後，未來要使用時就可以呼叫 keras.models.load_model 直接載入之前訓練好的模型



model_recovery = keras.models.load_model('SimpleRNN.h5')




dir(model_recovery)




model_recovery.metrics_names




type(model_recovery.metrics)




help(model_recovery.get_losses_for)




type(model_recovery.fit)


# - 儲存history (https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history)



import pickle
import codecs
with codecs.open('SimpleRNN_history.txt', 'wb', encoding='utf-8', errors='ignore') as f:
#with open('SimpleRNN_history.txt', 'wb') as f:
        pickle.dump(history.history, f)


# - 因為每篇評論只考慮前500個詞，而非完整評論！取得接近85%核驗集正確率仍屬不易，況且`SimpleRNN`不擅長處理長的有序數據，例如文本數據。
# - 接著介紹進階的遞歸類神經網路。

# ## 長短期記憶的LSTM類神經網路

# - 除了SimpleRNN()外，Keras尚有LSTM()與GRU()遞歸式結構，通常需要在此二結構中擇一建模。
# - GRU(Gated Recurrent Unit)的原理與LSTM相同，可說是流線型的LSTM，其學習成本較低，但知識表達能力相對較差。
# - LSTM將訊息儲存起來，防止較古老的訊息在處理的過程逐漸消逝(此即前述的vanishing gradient problem)。
# - LSTM的結構中除了有攜帶跨時間戳記訊息的額外數據流(carry track)，每個時間點並有計算新攜帶訊息的機制(根據投入訊息、輸出訊息與攜帶訊息進行計算)。
# - 簡而言之，LSTM讓過往的訊息在稍後的時點重新注入神經元，以解決vanishing gradient problem。

# ![](_img/AnatomyOfLSTM.png)

# ## IMDB LSTM建模示例

# - IMDB電影評論資料配適單一LSTM層，除了輸出層的維數外，其餘都依照Keras的預設值。



from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)




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


# ## 結論
# 
# - 在未調詞嵌入層維數或LSTM輸出維數，以及沒有模型係數正規化(regularization)的情況下，LSTM沒有取得突破性的進步。究其原因應是短評情感分析(sentiment analysis)並不屬於全域、長期結構的(global and long-term structure)分析。
# - 短評情感分析這個問題，基本上可藉由檢視每篇評論中出現的字詞，及其頻率來解析情感方向，這正是稠密層或完全連通網路的做法。
# - 在自然語言處理領域有更為困難的問題，此時LSTM正可發揮其所長，例如：問題回答與機器翻譯等。

# ## Reference: 
# 
# - CHOLLET, FRANÇOIS (2018), Deep Learning with Python, Manning.
# - Pal, A. and Prakash, PKS (2017), Practical Time Series Analysis: Master Time Series Data Processing, Visualization, and Modeling using Python, PACKT.
# - Raschka, S. (2015), Python Machine Learning: Unlock deeper insights into machine learning with this vital guide to cutting-edge predictive analytics, PACKT.
# - Ketkar, N. (2017), Deep Learning with Python: A Hands-on Introduction, Apress.

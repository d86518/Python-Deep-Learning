########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.


# coding: utf-8

# ## PM 2.5 時間序列預測 PM 2.5 Time Series Forecasting

# ### 資料理解與探索(Data Understanding and Exploration)
# 
# - PM 2.5 空汙時間序列預測模型
# - 直徑小於或等於2.5 micrometer的微粒(particulate matter)
# - 資料集可從UCI機器學習網站中下載(https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
#get_ipython().magic('matplotlib inline')
#help(get_ipython)
#dir(get_ipython())
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
# Python提供了future模組，把下一個新版本的特性導入到當前版本，於是我們就可以在當前版本中測試一些新版本的特性。 (https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001386820023084e5263fe54fde4e4e8616597058cc4ba1000)

#set current working directory
os.chdir('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes/')


#Read the dataset into a pandas.DataFrame
df = pd.read_csv('data/PRSA_data_2010.1.1-2014.12.31.csv')


print('Shape of the dataframe:', df.shape)

# > 變數定義與型別
# 
# - No: row number 
# - year: year of data in this row 
# - month: month of data in this row 
# - day: day of data in this row 
# - hour: hour of data in this row 
# - pm2.5: PM2.5 concentration (ug/m^3 microgram) 
# - DEWP: Dew Point (â„ƒ) 
# - TEMP: Temperature (â„ƒ) 
# - PRES: Pressure (hPa) 
# - cbwd: Combined wind direction 
# - Iws: Cumulated wind speed (m/s) 
# - Is: Cumulated hours of snow 
# - Ir: Cumulated hours of rain 

df.dtypes


# - 時資料


#Let's see the first five rows of the DataFrame
df.head(n=26)


help(df.head)

# pm2.5有遺缺值，cbwd為類別變數
df.describe(include="all") # Data of five years


df['cbwd'].value_counts()


"""
Rows having NaN values in column pm2.5 are dropped.
"""
df.dropna(subset=['pm2.5'], axis=0, inplace=True) # df.shape change to (41757,13) from (43824, 13)

print(df.index)

df.reset_index(drop=True, inplace=True)


print('Shape of the dataframe:', df.shape) # 43824 - 41757 = 2067 NAs


# > 資料前處理與視覺化
# 
# - 運用datetime.datetime函數產生新欄位datetime，並依此新欄位排序整個資料框 (To make sure that the rows are in the right order of date and time of observations, a new column datetime is created from the date and time related columns of the DataFrame. The new column consists of Python's datetime.datetime objects. The DataFrame is sorted in ascending order over this column.)
# - Append a new column in Pandas DataFrame is so easy !

help(df.apply)

df['datetime'] = df[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'], month=row['month'], day=row['day'], hour=row['hour']), axis=1)

print(df.dtypes['datetime']) # datetime: datetime64[ns]

df.sort_values(by='datetime', ascending=True, inplace=True)


# - 視覺化檢視pm2.5的離群值

# A box plot for pm2.5 is plotted to check the presence of outliers
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['pm2.5'], width=0.3)
g.set_title('Box plot of pm2.5')


# - 因有離群值，故以MAE為人工神經網路權重訓練之優化準則(較不易受離群值的影響)

# - 視覺化檢視Pressure

#Let us draw a box plot to visualize the central tendency and dispersion of PRES
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['PRES'], width=0.3)
g.set_title('Box plot of Air Pressure')


# - 檢視整個資料期間的pm2.5時間序列，重點在是否有趨勢、季節性等等特徵
# - 沒有長期趨勢

#import matplotlib.dates as mdates
#datetime = df['datetime']
#df.append({'Date': mdates.date2num(datetime)})

plt.figure(figsize=(11, 11))
g = sns.tsplot(df['pm2.5'], time=df["datetime"])
g.set_title('Time series of pm2.5')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


# - 六個月期間的時間序列圖似乎有週期性高峰與低谷，但相鄰兩高峰或低谷的間隔時間不定
# - 高峰的高度亦呈現相當程度的波動
# - 一個大高峰旁邊會伴隨幾個較矮的高峰

#Let's plot the series for six months to check if any pattern apparently exists.
plt.figure(figsize=(11, 11))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=6,day=30)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')

#Let's plot the series for twelve months to check if any pattern apparently exists.
plt.figure(figsize=(11, 11))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=12,day=31)], color='g')
g.set_title('pm2.5 during 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')

#Let's zoom in on one month.
plt.figure(figsize=(11, 11))
g = sns.tsplot(df['pm2.5'].loc[df['datetime']<=datetime.datetime(year=2010,month=1,day=31)], color='g')
g.set_title('pm2.5 during Jan 2010')
g.set_xlabel('Index')
g.set_ylabel('pm2.5 readings')


# - 檢視整個資料期間的PRES時間序列

plt.figure(figsize=(11, 11))
g = sns.tsplot(df['PRES'])
g.set_title('Time series of Air Pressure')
g.set_xlabel('Index')
g.set_ylabel('Air Pressure readings in hPa')


# - 最陡坡降演算法當變數在[-1, 1]，或是[-3, 3]時表現比較好(收斂較快！)
# - 以MinMaxScaler將pm2.5和PRES轉換到[0,1]之間

df['pm2.5'].head()

from sklearn.preprocessing import MinMaxScaler
scaler_pm25 = MinMaxScaler(feature_range=(0, 1))
df['scaled_pm2.5'] = scaler_pm25.fit_transform(np.array(df['pm2.5']).reshape(-1, 1)) # Why reshape?

# - newshape : int or tuple of ints. The new shape should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

df['scaled_pm2.5'].head()

# - 為何要reshape?類似R語言之drop引數設為TRUE或是FALSE。

np.array(df['pm2.5']).shape

np.array(df['pm2.5']).reshape(-1, 1).shape

df['PRES'].head()

scaler_pres = MinMaxScaler(feature_range=(0, 1))
df['scaled_PRES'] = scaler_pres.fit_transform(np.array(df['PRES']).reshape(-1, 1))

df['scaled_PRES'].head()

# > 資料切分
# 
# - 依訓練集計算損失函數，以最陡坡降演算法進行誤差倒傳遞與權重更新
# - 核驗集用來評估模型與決定最佳訓練代數(epoch)，增加代數可進一步降低損失函數值，但可能招致過度配適風險
# - 前端Keras + 後端TensorFlow

"""
Let's start by splitting the dataset into train and validation. The dataset's time period if from
Jan 1st, 2010 to Dec 31st, 2014. The first fours years - 2010 to 2013 is used as train and
2014 is kept for validation.
"""
split_date = datetime.datetime(year=2014, month=1, day=1, hour=0)
df_train = df.loc[df['datetime']<split_date]
df_val = df.loc[df['datetime']>=split_date]
print('Shape of train:', df_train.shape)
print('Shape of validation:', df_val.shape)


# - 檢視訓練集

#First five rows of train
df_train.head()


# - 檢視核驗集

#First five rows of validation
df_val.head()


# - 重設核驗集的觀測值索引(或編號)

#Reset the indices of the validation set
df_val.reset_index(drop=True, inplace=True) # 0, 1, 2, ...


# - 繪製訓練集與核驗集的scaled_pm2.5時間序列折線圖

"""
The train and validation time series of scaled pm2.5 is also plotted.
"""

plt.figure(figsize=(11, 11))
g = sns.tsplot(df_train['scaled_pm2.5'], color='b')
g.set_title('Time series of scaled pm2.5 in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')

plt.figure(figsize=(11, 11))
g = sns.tsplot(df_val['scaled_pm2.5'], color='r')
g.set_title('Time series of scaled pm2.5 in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled pm2.5 readings')


# - 也繪出訓練集與核驗集的scaled_PRES時間序列折線圖

"""
The train and validation time series of standardized PRES are also plotted.
"""

plt.figure(figsize=(11, 11))
g = sns.tsplot(df_train['scaled_PRES'], color='b')
g.set_title('Time series of scaled Air Pressure in train set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')
# plt.savefig('plot/B07887_05_03.png', format='png', dpi=300)

plt.figure(figsize=(11, 11))
g = sns.tsplot(df_val['scaled_PRES'], color='r')
g.set_title('Time series of scaled Air Pressure in validation set')
g.set_xlabel('Index')
g.set_ylabel('Scaled Air Pressure readings')
# plt.savefig('plot/B07887_05_04.png', format='png', dpi=300)


# > 產生訓練集與核驗集的X與y
# 
# - 用過去七天的觀測值來預測下一天的pm2.5值，亦即是AR(7)模型
# - makeXy(.)函數依傳入的原始時間序列，以及所需的歷史觀測值天數，生成X與y

def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors (predictors)
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]): # start = 7 ~ stop = 33095
        X.append(list(ts.loc[i-nb_timesteps:i-1])) # 0~6, 1~7, 2~8, ...
        y.append(ts.loc[i]) # 7, 8, 9, ...
    X, y = np.array(X), np.array(y)
    return X, y

X_train, y_train = makeXy(df_train['scaled_pm2.5'], 7)
print('Shape of train arrays:', X_train.shape, y_train.shape)

X_train_press, y_train_press = makeXy(df_train['scaled_PRES'], 7)
print('Shape of train arrays:', X_train_press.shape, y_train_press.shape)

X_val, y_val = makeXy(df_val['scaled_pm2.5'], 7)
print('Shape of validation arrays:', X_val.shape, y_val.shape)

X_val_press, y_val_press = makeXy(df_val['scaled_PRES'], 7)
print('Shape of validation arrays:', X_val_press.shape, y_val_press.shape)
# 注意！y_val與y_val_press擷取scaled後的變數

### 多層感知機(Multi-Layer Preceptron, MLP)

from keras.layers import Dense, Input, Dropout # 投入層、稠密層、丟棄層
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,), dtype='float32')

#Dense layers are defined with linear activation
dense1 = Dense(32, activation='tanh')(input_layer)
dense2 = Dense(16, activation='tanh')(dense1)
dense3 = Dense(16, activation='tanh')(dense2)


# Multiple hidden layers and large number of neurons in each hidden layer gives neural networks the ability to model complex non-linearity of the underlying relations between regressors and target. However, deep neural networks can also overfit train data and give poor results on validation or test set. Dropout has been effectively used to regularize deep neural networks. In this example, a Dropout layer is added before the output layer. Dropout randomly sets p fraction of input neurons to zero before passing to the next layer. Randomly dropping inputs essentially acts as a bootstrap aggregating or bagging type of model ensembling. Random forest uses bagging by building trees on random subsets of input features. We use p=0.2 to dropout 20% of randomly selected input features.

dropout_layer = Dropout(0.2)(dense3)

#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# The input, dense and output layers will now be packed inside a Model, which is wrapper class for training and making predictions. The box plot of pm2.5 shows the presence of outliers. Hence, mean absolute error (MAE) is used as absolute deviations suffer less fluctuations compared to squared deviations.
# 
# The network's weights are optimized by the Adam algorithm. Adam stands for adaptive moment estimation and has been a popular choice for training deep neural networks. Unlike, stochastic gradient descent, adam uses different learning rates for each weight and separately updates the same as the training progresses. The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients.

ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')
ts_model.summary()


# The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be used for a instance of back propagation.The validation dataset is also passed to evaluate the model after every epoch completes. A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch, at which the loss function has been minimum.

# > PM2.5多層感知機
save_weights_at = os.path.join('data', 'PRSA_data_PM2.5_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)

ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)

# > PRES多層感知機
save_weights_at_2 = os.path.join('data', 'PRSA_data_Air_Pressure_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best_2 = ModelCheckpoint(save_weights_at_2, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)

ts_model.fit(x=X_train_press, y=y_train_press, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best_2], validation_data=(X_val_press, y_val_press),
             shuffle=True)


# Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the scaled pm2.5, are inverse transformed to get predictions of original pm2.5.

#os.listdir("data/")
best_model = load_model(os.path.join('data', 'PRSA_data_PM2.5_MLP_weights.20-0.0120.hdf5')) # 請自行檢視檔名，每個人可能都不同。如果路徑檔名有誤，1st epoch訓練完後就會停止！
preds = best_model.predict(X_val)
#preds.min()
#preds.max()

###############
#plt.figure(figsize=(11, 11))
#plt.plot(range(50), y_val[:50], linestyle='-', marker='*', color='r')
#plt.plot(range(50), preds[:50], linestyle='-', marker='.', color='b')
#plt.legend(['Actual','Predicted'], loc=2)
#plt.title('Actual vs Predicted pm2.5')
#plt.ylabel('pm2.5')
#plt.xlabel('Index')
###############


pred_pm25 = scaler_pm25.inverse_transform(preds)
#pred_pm25.min()
#pred_pm25.max()
#y_val.min()
#y_val.max()

pred_pm25 = np.squeeze(pred_pm25) # The input array, but with all or a subset of the dimensions of length 1 removed. (8654, 1) -> (8654,)

#os.listdir("data/")
best_model_PRES = load_model(os.path.join('data', 'PRSA_data_Air_Pressure_MLP_weights.20-0.0089.hdf5'))
preds = best_model_PRES.predict(X_val_press)
pred_PRES = scaler_pres.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))

r2 = r2_score(df_val['PRES'].loc[7:], pred_PRES)
print('R-squared for the validation set:', round(r2,4))

#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')
# plt.savefig('plot/B07887_05_09.png', format='png', dpi=300)

#Let's plot the first 50 actual and predicted values of air pressure.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['PRES'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('Air Pressure')
plt.xlabel('Index')
# plt.savefig('plot/B07887_05_05.png', format='png', dpi=300)


# > 1D卷積類神經網路(Convolution Neural Neworks)
# 
# - 卷積層投入的資料維度為3D，其維數為(樣本數, 延遲期數, 每期屬性數)
# - 本例為(33089或8654, 7, 1)
# - 每期屬性數為1，因此為1D卷積層


#X_train and X_val are reshaped to 3D arrays
X_train, X_val = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),                 X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print('Shape of arrays after reshaping:', X_train.shape, X_val.shape)


# > Keras API定義CNN
# 
# - 定義下一層時，再宣告其投入層


from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


# - 投入層


#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(7,1), dtype='float32')
input_layer

# - 0補綴層：前後填補0，使得卷積層運算出來的結果與原時間序列長度一樣(ZeroPadding1D layer is added next to add zeros at the begining and end of each series. Zeropadding ensure that the downstream convolution layer does not reduce the dimension of the output sequences.)
# - 卷積層與跨度(stride)
# - 合併層(Pooling layer, 又稱降低取樣層downsampling layer)：提取卷積層運算出來的結果


#Add zero padding
zeropadding_layer = ZeroPadding1D(padding=1)(input_layer)


# - (輸出屬性數, 1D卷積時窗長度, 平移跨度, 是否使用截距項)(The first argument of Conv1D is the number of filters, which determine the number of features in the output. Second argument indicates length of the 1D convolution window. The third argument is strides and represent the number of places to shift the convolution window. Lastly, setting use_bias as True, add a bias value during computation of an output feature. Here, the 1D convolution can be thought of as generating local AR models over rolling window of three time units.)


#Add 1D convolution layers
conv1D_layer1 = Conv1D(64, 3, strides=1, use_bias=True)(zeropadding_layer)
conv1D_layer2 = Conv1D(32, 3, strides=1, use_bias=True)(conv1D_layer1)


# - 合併層以平均合併，而非以最大值合併(AveragePooling1D is added next to downsample the input by taking average over pool size of three with stride of one timesteps. The average pooling in this case can be thought of as taking moving averages over a rolling window of three time units. We have used average pooling instead of max pooling to generate the moving averages.)


#Add AveragePooling1D layer
avgpooling_layer = AveragePooling1D(pool_size=3, strides=1)(conv1D_layer2)


# - 合併後產生3D輸出，因此需要扁平化層(樣本數, 延遲期數*每期屬性數)(The preceeding pooling layer returns 3D output. Hence before passing to the output layer, a Flatten layer is added. The Flatten layer reshapes the input to (number of samples, number of timesteps*number of features per timestep), which is then fed to the output layer)


#Add Flatten layer
flatten_layer = Flatten()(avgpooling_layer)


#A couple of Dense layers are also added
dense_layer1 = Dense(32)(avgpooling_layer)
dense_layer2 = Dense(16)(dense_layer1)


# - 丟棄層(dropout layer)：因深度類經網路容易過度配適訓練集，導致驗證或測試集績效不彰。因此，在輸出層前加入一丟棄層，隨機將p百分比的投入值設為0，獲得拔靴整合或裝袋型系集模型的過度配適矯正效果

dropout_layer = Dropout(0.2)(flatten_layer)

#Finally the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='linear')(dropout_layer)


# - 打包投入層、隱藏層與輸出層為為模型
# - 以MAE為優化準則
# - adam優化算法


ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_absolute_error', optimizer='adam')#SGD(lr=0.001, decay=1e-5))
ts_model.summary()


# - 模型配適 (The model is trained by calling the fit function on the model object and passing the X_train and y_train. The training is done for a predefined number of epochs. Additionally, batch_size defines the number of samples of train set to be used for a instance of back propagation. The validation dataset is also passed to evaluate the model after every epoch completes.)
# - 以核驗集追蹤損失函數，並儲存各epoch的最小損失值結果 (A ModelCheckpoint object tracks the loss function on the validation set and saves the model for the epoch, at which the loss function has been minimum.)


save_weights_at = os.path.join('data', 'PRSA_data_PM2.5_1DConv_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)
ts_model.fit(x=X_train, y=y_train, batch_size=16, epochs=20,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)



help(ModelCheckpoint)

X_train


y_train


# - 以儲存的最佳模型進行預測，因預測值已標準化，故須逆轉換回原始pm2.5 (Prediction are made for the pm2.5 from the best saved model. The model's predictions, which are on the standardized pm2.5, are inverse transformed to get predictions of original pm2.5.)
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#os.listdir()
best_model = load_model('data/PRSA_data_PM2.5_1DConv_weights.16-0.0130.hdf5')
preds = best_model.predict(X_val)
pred_pm25 = scaler_pm25.inverse_transform(preds)
pred_pm25 = np.squeeze(pred_pm25)


from sklearn.metrics import mean_absolute_error



mae = mean_absolute_error(df_val['pm2.5'].loc[7:], pred_pm25)
print('MAE for the validation set:', round(mae, 4))


#Let's plot the first 50 actual and predicted values of pm2.5.
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(50), df_val['pm2.5'].loc[7:56], linestyle='-', marker='*', color='r')
plt.plot(range(50), pred_pm25[:50], linestyle='-', marker='.', color='b')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted pm2.5')
plt.ylabel('pm2.5')
plt.xlabel('Index')


# ## 參考文獻
# 
# - Pal, A. and Prakash, P. (2017), Practical Time Series Analysis: Master Time Series Data Processing, Visualization, and Modeling using Python, Packt.

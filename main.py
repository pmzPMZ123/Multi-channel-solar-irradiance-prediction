# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 0010 15:21
# @Author  : pimaozheng
# @Site    : 
# @File    : model.py
# @Software: PyCharmim
from pyhht.visualization import plot_imfs
from newSWT import *
from sklearn import ensemble  #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors  #### KNN回归####
from sklearn.neural_network import MLPRegressor    #### MLP 感知机####
from sklearn import svm                            #### SVM回归####
#from lstmRegressor import lstmRegressor  #### lstm回归
from keras.initializers import he_normal
from sklearn.metrics import r2_score
from  keras.layers import LeakyReLU,Concatenate,Multiply,Bidirectional
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional, Flatten, Add, Concatenate, MaxPool1D, LeakyReLU
import pandas as pd
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from evaluate_data import *
from pyhht.emd import EMD
import tensorflow as tf
import xlwt
import numpy as np
from numpy import *
#import lightgbm as lgbm
#import xgboost
import warnings
#from newSWT import *
from keras.models import *
from keras.layers import merge
from keras.layers.core import *
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense,LSTM,MaxPooling1D,Dropout,AveragePooling1D
from keras.layers.convolutional import Conv1D
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from nested_lstm import NestedLSTM
import keras


def smape(y_true, y_pred):
    L1 = int(len(y_true))
    L2 = int(len(y_pred))
    # print(L1,L2)
    if L1 == L2:
        # SUM1=sum(abs(true-predict)/abs(true))
        SUM = 0.0
        for i in range(L1):
            a= 2.0*(abs(y_true[i] - y_pred[i]))
            b=np.abs(y_pred[i]) + np.abs(y_true[i])
            SUM=a/b+SUM
        per_SUM = SUM * 100.0
        smape = per_SUM / L1
        return smape

#构建数据
def create_data0(data, train_num, ahead_num):
    dataX1, dataX2 = [], []
    dataY1, dataY2 = [], []

    for i in range(train_num - ahead_num):
        # print(i)
        a = data[i:(i + ahead_num), :2]
        dataX1.append(a)
    for j in range(train_num - ahead_num, len(data) - ahead_num):
        b = data[j:(j + ahead_num), :2]
        dataX2.append(b)

    dataY1 = data[ahead_num:train_num, 0]
    dataY2 = data[train_num:, 0]
    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)
def Create_dataset(dataset,look_back):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1 ):
        a = dataset[i:(i + look_back)]
        data_X.append(a)
        data_Y.append(dataset[i + look_back])
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return  data_X,data_Y
def create_time_series(data, time_step):
    train_num = len(data)
    TS_X = []

    for i in range(train_num - time_step):
        b = data[i:(i + time_step), 0]
        TS_X.append(b)

    TS_X = np.array(TS_X)
    return TS_X
def load_data_ts(trainNum, testNum, startNum, data):
    print('General_data loading.')

    global ahead_num
    ahead_num=8
    # all_data_checked = data

    targetData = data

    global x_mode

    time_series_y = create_time_series(targetData, ahead_num)

    allX = np.c_[time_series_y]

    allX = allX.T
    print("\nallX:", allX.shape)

    ###########======================================

    trainX = allX[:, : trainNum]
    trainY = targetData.T[:, ahead_num: trainNum + ahead_num]
    testX = allX[:, trainNum:]
    testY = targetData.T[:, trainNum + ahead_num: (trainNum + testNum)]

    trainY = trainY.flatten()  # 降维
    testY = testY.flatten()  # 降维
    trainX = trainX.T
    testX = testX.T

    print('load_data complete.\n')

    return trainX, trainY, testX, testY

def create_data(data, train_num, time_step):
    TS_X = []

    for i in range(data.shape[0] - time_step):
        b = data[i:(i + time_step), 0]
        TS_X.append(b)


    dataX1 = TS_X[:train_num]
    dataX2 = TS_X[train_num:]
    dataY1 = data[time_step: train_num + time_step, 0]
    dataY2 = data[train_num + time_step:, 0]

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)




def load_data_wvlt(trainNum, testNum, startNum, data):
    print('wavelet_data loading.')

    global ahead_num
    # all_data_checked = data
    targetData = data

    testY = None
    global wvlt_lv
    wvlt_lv=3
    wavefun = pywt.Wavelet('db1')
    coeffs = swt_decom(targetData, wavefun, wvlt_lv)

    ### 测试滤波效果
    wvlt_level_list = []
    for wvlt_level in range(len(coeffs)):
        wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY = create_data(coeffs[wvlt_level], trainNum, ahead_num)
        wvlt_level_part = [wvlt_trainX, wvlt_trainY, wvlt_testX, wvlt_testY]
        wvlt_level_list.append(wvlt_level_part)

    print('load_data complete.\n')

    return wvlt_level_list, testY
def load_data(filename, ahead_num):

    dataset = pd.read_csv(filename, encoding='gbk')
    #加载ghi数据
    dataPOV = dataset.iloc[0:,0]
    #加载temp数据
    dataAir = dataset.iloc[0:, 1]

    N_train =40000
    N_test = 7000
    startNum = 1
    ahead_num = 8
    N2 = N_train + N_test
    #分别归一化
    global scaler_POV
    dataPOV = np.array(dataPOV[startNum + 1: startNum +N_train + N_test + 1]).reshape(-1, 1)
    scaler_POV = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_POV = scaler_POV.fit_transform(dataPOV)

    global scaler_Air
    dataAir = np.array(dataAir[startNum + 1: startNum + N_train + N_test + 1]).reshape(-1, 1)
    scaler_Air = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataAir = scaler_Air.fit_transform(dataAir)

    x_train, y_train, x_test, y_test = load_data_ts(N_train, N_test,  startNum, dataPOV)
    x_train1, y_train1, x_test1, y_test1 = load_data_ts(N_train, N_test, startNum, dataAir)

    train_x_list = [x_train.reshape(-1, ahead_num, 1),
                    x_train1.reshape(-1, ahead_num, 1)
                    ]

    test_x_list = [x_test.reshape(-1, ahead_num, 1),
                   x_test1.reshape(-1, ahead_num, 1)
                   ]
    train_y_list = [y_train, y_train1]
    test_y_list = [y_test, y_test1]

    wvlt_list, _ = load_data_wvlt(N_train, N_test,  startNum, dataPOV)
    wvlt_list1, _ = load_data_wvlt(N_train, N_test,  startNum, dataAir)

    wvlt_trX_list = []
    wvlt_teX_list = []
    for i_wvlt in range(len(wvlt_list)):
        wvlt_trX = np.reshape(wvlt_list[i_wvlt][0],
                              (wvlt_list[i_wvlt][0].shape[0],
                               wvlt_list[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list[i_wvlt][2],
                              (wvlt_list[i_wvlt][2].shape[0],
                               wvlt_list[i_wvlt][2].shape[1], 1))
        wvlt_trX_list.append(wvlt_trX)
        wvlt_teX_list.append(wvlt_teX)

    wvlt_trX_list1 = []
    wvlt_teX_list1 = []
    for i_wvlt in range(len(wvlt_list1)):
        wvlt_trX = np.reshape(wvlt_list1[i_wvlt][0],
                              (wvlt_list1[i_wvlt][0].shape[0],
                               wvlt_list1[i_wvlt][0].shape[1], 1))
        wvlt_teX = np.reshape(wvlt_list1[i_wvlt][2],
                              (wvlt_list1[i_wvlt][2].shape[0],
                               wvlt_list1[i_wvlt][2].shape[1], 1))
        wvlt_trX_list1.append(wvlt_trX)
        wvlt_teX_list1.append(wvlt_teX)
    wvlt_list_train_MV = wvlt_trX_list + \
                         wvlt_trX_list1

    wvlt_list_test_MV = wvlt_teX_list + \
                        wvlt_teX_list1

    model0 = build_Multimodel(ahead_num)


    print("==================")
    print( test_y_list[0])
    history = model0.fit(wvlt_list_train_MV, train_y_list, epochs=16, validation_split=0.05, verbose=1
                    )
    predict_list = model0.predict( wvlt_list_test_MV)
    print("Saving model to disk \n")
    predict_Pov = predict_list[0]
    predict_Pov = predict_Pov.reshape(-1, )
    test_y_pov= test_y_list[0]



    rmse_WN = RMSE1(test_y_pov, predict_Pov)
    print("rmse", rmse_WN)
    mape_WN = MAPE1(test_y_pov, predict_Pov)
    print("mape", mape_WN)
    mae_WN = MAE1(test_y_pov, predict_Pov)
    print("mae", mae_WN)
    mse_WN = MSE1(test_y_pov, predict_Pov)
    print("mse", mse_WN)
    r2_wn = r2_score(test_y_pov, predict_Pov)
    print("R2", r2_wn)
    smape3 = smape( test_y_pov, predict_Pov)
    print("duibi3:", smape3)

    plt.figure(1, figsize=(10, 6))
    plt.plot(test_y_pov[:], "black", label="True", linewidth=2)
   # plt.plot(predict_POV_n[:], 'red', label='LSTM', linewidth=1.75, linestyle='--')
    plt.plot(predict_Pov[:], 'r', label='Proposed', linewidth=2)
    plt.legend(loc='upper right')

    plt.grid()
    plt.show()









def build_Multimodel(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    ##############################################################################################

    input_0A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_0A3)
    x00 =Bidirectional(LSTM(100, return_sequences=False))(con1)
    x00 = Dense(8, activation='linear')(x00)

    input_0D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_0D1)
    x01 = Bidirectional(LSTM(100, return_sequences=False))(con1)
    x01 = Dense(8, activation='linear')(x01)

    input_0D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_0D2)
    x02 =Bidirectional(LSTM(64, return_sequences=False))(con1)
    x02 = Dense(8, activation='linear')(x02)

    input_0D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x03 = Bidirectional(LSTM(32, return_sequences=False))(input_0D3)
    x03 = Dense(8, activation='linear')(x03)

    combined0 = Add()([x00, x01, x02, x03])
    o0 = combined0
    # o0 = Dense(1, activation="linear")(combined0)

    model0 = Model(inputs=[input_0A3, input_0D1, input_0D2,input_0D3], outputs=o0)

    ##############################################################################################

    input_1A3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_1A3)
    x10 = Bidirectional(LSTM(100, return_sequences=False))(con1)
    x10 = Dense(8, activation='linear')(x10)

    input_1D1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_1D1)
    x11 = Bidirectional(LSTM(100, return_sequences=False))(con1)
    x11 = Dense(8, activation='linear')(x11)

    input_1D2 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_1D2)
    x12 = Bidirectional(LSTM(64, return_sequences=False))(con1)
    x12 = Dense(8, activation='linear')(x12)

    input_1D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_1D3)
    x13 = Bidirectional(LSTM(32, return_sequences=False))(con1)
    x13 = Dense(8, activation='linear')(x13)

    combined1 = Add()([x10, x11, x12,x13])
    o1 = combined1
    # o1 = Dense(1, activation="linear")(combined1)

    model1 = Model(inputs=[input_1A3, input_1D1, input_1D2,input_1D3], outputs=o1)

    ##############################################################################################


    ##############################################################################################

    combined = Concatenate(axis=1)([o0, o1])

    Output0 = Dense(1, activation='linear')(combined)

    Output1 = Dense(1, activation='linear')(combined)



    model = Model(inputs=[input_0A3, input_0D1, input_0D2,input_0D3,
                          input_1A3, input_1D1, input_1D2,input_1D3
                          ],
                  outputs=[Output0,
                           Output1
                           ])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model

#Multiv-MC-WT-Bilstm
def buildNLSTM_MultiV_v5(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    ##############################################################################################

    input_0A3 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x00 = Bidirectional(LSTM(100, return_sequences=False))(input_0A3)
    x00 = Dense(8, activation='linear')(x00)

    input_0D1 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x01 = Bidirectional(LSTM(100, return_sequences=False))(input_0D1)
    x01 = Dense(8, activation='linear')(x01)

    input_0D2 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x02 = Bidirectional(LSTM(64, return_sequences=False))( input_0D2 )
    x02 = Dense(8, activation='linear')(x02)

    input_0D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x03 = Bidirectional(LSTM(32, return_sequences=False))(input_0D3)
    x03 = Dense(8, activation='linear')(x03)

    combined0 = Add()([x00, x01, x02,x03])
    o0 = combined0
    # o0 = Dense(1, activation="linear")(combined0)

    model0 = Model(inputs=[input_0A3, input_0D1, input_0D2,input_0D3], outputs=o0)

    ##############################################################################################

    input_1A3 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x10 = Bidirectional(LSTM(100, return_sequences=False))(input_1A3)
    x10 = Dense(8, activation='linear')(x10)

    input_1D1 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x11 = Bidirectional(LSTM(64, return_sequences=False))(input_1D1)
    x11 = Dense(8, activation='linear')(x11)

    input_1D2 = Input(batch_shape=(batch_size, timesteps, input_dim))

    x12 =Bidirectional(LSTM(32, return_sequences=False))(input_1D2)
    x12 = Dense(8, activation='linear')(x12)

    input_1D3 = Input(batch_shape=(batch_size, timesteps, input_dim))
    x13 = Bidirectional(LSTM(32, return_sequences=False))(input_1D3)
    x13 = Dense(8, activation='linear')(x13)

    combined1 = Add()([x10, x11, x12,x13])
    o1 = combined1
    # o1 = Dense(1, activation="linear")(combined1)

    model1 = Model(inputs=[input_1A3, input_1D1, input_1D2,input_1D3], outputs=o1)

    ##############################################################################################


    ##############################################################################################

    combined = Concatenate(axis=1)([o0, o1])

    Output0 = Dense(1, activation='linear')(combined)

    Output1 = Dense(1, activation='linear')(combined)



    model = Model(inputs=[input_0A3, input_0D1, input_0D2,input_0D3,
                          input_1A3, input_1D1, input_1D2,input_1D3
                          ],
                  outputs=[Output0,
                           Output1
                           ])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model

def buildNLSTM_MultiV_v2(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1

    input_0 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_0)
    x0 = Bidirectional(LSTM(100, return_sequences=False))(con1)
    x0 = Dense(16, activation='linear')(x0)
    model0 = Model(inputs=input_0, outputs=x0)

    input_1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    con1 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, 1))(input_1)
    x1 = Bidirectional(LSTM(100, return_sequences=False))(con1)
    x1 = Dense(16, activation='linear')(x1)
    model1 = Model(inputs=input_1, outputs=x1)

    combined = Concatenate(axis=1)([model0.output,
                                    model1.output,
                                    ])

    output0 = Dense(1, activation='linear')(combined)

    output1 = Dense(1, activation='linear')(combined)



    model = Model(inputs=[model0.input,
                          model1.input],
                  outputs=[output0,
                           output1]
                  )

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model

def multi_head_cnn_model( train_x, train_y):
    '''
    该函数定义 Multi-head CNN 模型
    '''
    #train_x, train_y = sliding_window(train, sw_width)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    in_layers, out_layers = [], []  # 用于存放每个特征序列的CNN子模型
    for i in range(n_features):
        inputs = Input(shape=(n_timesteps, 1))

        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        x   = NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0.0,)(pool1)
        flat = Flatten()(x)
        in_layers.append(inputs)
        out_layers.append(flat)

    merged = concatenate(out_layers)  # 合并八个CNN子模型

    dense1 = Dense(200, activation='relu')(merged)  # 全连接层对上一层输出特征进行解释
    dense2 = Dense(100, activation='relu')(dense1)
    outputs = Dense(n_outputs)(dense2)
    model = Model(inputs=in_layers, outputs=outputs)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()

   # plot_model(model, to_file='multi-head-cnn-energy-usage-prediction.png', show_shapes=True, show_layer_names=True,
  #             dpi=300)

 #   input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]

    # 这里只是为了方便演示和输出loss曲线，不建议这么做，这样其实是训练了2倍的epoch；
    # 可以保存模型，再加载预测；或者直接将预测函数定影在这里，省去调用步骤。
 #   model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
#    history = model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model
def build_multi_cr_lstm_model(ts, fea_dim):
    # 定义输入
    batch_size, timesteps, input_dim = None, ts, 1
    #inputs = Input(shape=(ts, fea_dim))
    inputs= Input(batch_shape=(batch_size, ts, fea_dim))
    # ########################################
    # cnn层&lstm层1
    cnn_left_out1 = Conv1D(filters=32, kernel_size=3, strides=3, kernel_initializer=he_normal(seed=3))(inputs)
    act_left_out1 = LeakyReLU()(cnn_left_out1)

    lstm_left_out1 = NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0.0,)(act_left_out1)

    # #########################################
    # cnn层&lstm层2
    cnn_right_out1 = Conv1D(filters=32, kernel_size=5, strides=3, kernel_initializer=he_normal(seed=3))(inputs)
    act_right_out1 = LeakyReLU()(cnn_right_out1)

    lstm_right_out1 = NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0.0,)(act_right_out1)

    # #########################################
    # cnn层&lstm层3
    cnn_mid_out1 = Conv1D(filters=32, kernel_size=2, strides=3, kernel_initializer=he_normal(seed=3))(inputs)
    act_mid_out1 = LeakyReLU()(cnn_mid_out1)

    lstm_mid_out1 = NestedLSTM(32, depth=2, dropout=0, recurrent_dropout=0.0,)(act_mid_out1)

    # ############################################
    # 上层叠加新的dense层
    concat_output = Concatenate(axis=1)([lstm_left_out1, lstm_mid_out1, lstm_right_out1])
    outputs = Dense(1)(concat_output)
    model_func = Model(inputs=inputs, outputs=outputs)
    model_func.compile(loss='mse', optimizer=Adam(lr=0.02, decay=0.003), metrics=['mse'])




if __name__ == "__main__":
    filename="C:\\Users\\Administrator\\PycharmProjects\\solar\\GHI_irradiance_10min.csv"
    ahead_num = 8
    load_data(filename,ahead_num)
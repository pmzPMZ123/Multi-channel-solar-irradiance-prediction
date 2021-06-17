# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 0024 18:27
# @Author  : pimaozheng
# @Site    : 
# @File    : load_model.py
# @Software: PyCharm
from pyhht.visualization import plot_imfs
from newSWT import *
from sklearn.neural_network import MLPRegressor    #### MLP 感知机####
from sklearn.tree import ExtraTreeRegressor        #### ExtraTree极端随机树回归####
from sklearn import tree                           #### 决策树回归####
from sklearn.ensemble import BaggingRegressor      #### Bagging回归####
from sklearn.ensemble import AdaBoostRegressor     #### Adaboost回归
from sklearn import linear_model                   #### 线性回归####
from sklearn import svm                            #### SVM回归####
from sklearn import ensemble                       #### Adaboost回归####  ####3.7GBRT回归####  ####3.5随机森林回归####
from sklearn import neighbors                      #### KNN回归####

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
import csv


def skill(true_data, predict_data):
    testY = true_data
    testPredict = predict_data

    # rmse = mse ** 0.5
    rmse = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
    rmse_p=np.sqrt(np.mean((testY[:]-testPredict[:])**2))
    skill= skill = 1.0 - rmse / rmse_p
    return skill
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)
def MAPE2(true,predict):
    L1=int(true.shape[0])
    L2=int(predict.shape[0])
    print(L1)
    print(L2)
    #print(L1,L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1-1):
            SUM=(abs(true[i,0]-predict[i,0])/true[i,0])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")
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


    return 2.0 * np.mean(np.abs(y_pred[i] - y_true[i]) / (np.abs(y_pred[i]) + np.abs(y_true[i]))) * 100
#list 版
def MAPE1(true,predict):
#    L1=int(true.shape[0])
#    L2=int(predict.shape[0])
    L1 = int(len(true))
    L2 = int(len(predict))
    #print(L1,L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1):
            #SUM=(abs(true[i,0]-predict[i,0])/true[i,0])+SUM
            SUM = abs((true[i]-predict[i])/true[i])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")

def MSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mse = np.sum((predict_data-true_data)**2)/len(true_data) #跟数学公式一样的
    mse = mean_squared_error(testY[:,0], testPredict[:, 0])
    return mse

def MSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mse = mean_squared_error(testY[:], testPredict[:])
    return mse

def MBE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data

    mbe =np.nanmean(testY[:]-testPredict[:])
    return mbe
def RMSE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    # rmse = mse ** 0.5
    rmse = math.sqrt( mean_squared_error(testY[:,0], testPredict[:, 0]))
    return rmse

def RMSE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    rmse = math.sqrt( mean_squared_error(testY[:], testPredict[:]))
    return rmse


def MAE(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    #mae = np.sum(np.absolute(predict_data - true_data))/len(true_data)
    mae=mean_absolute_error(testY[:,0], testPredict[:, 0])
    return mae

def MAE1(true_data, predict_data):
    testY = true_data
    testPredict = predict_data
    mae=mean_absolute_error(testY[:], testPredict[:])
    return mae
def pre_model(model,trainX,trainY,testX):
    model.fit(trainX,trainY)
    predict = model.predict(testX)
    return predict

def create_data(data, train_num, ahead_num):
    dataX1, dataX2 = [], []
    dataY1, dataY2 = [], []

    for i in range(train_num - ahead_num):
        # print(i)
        a = data[i:(i + ahead_num), 0]
        dataX1.append(a)
    for j in range(train_num - ahead_num, len(data) - ahead_num):
        b = data[j:(j + ahead_num), 0]
        dataX2.append(b)

    dataY1 = data[ahead_num:train_num, 0]
    dataY2 = data[train_num:, 0]
    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)
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

    # print("allX:", allX.shape)
    # print("trainX:", trainX.shape)
    # print("trainY:", trainY.shape)
    # print("testX:", testX.shape)
    # print("testY:", testY.shape)

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

    # dataX1 = TS_X[:train_num]
    # dataX2 = TS_X[train_num:]
    # dataY1 = data[time_step: train_num + time_step, 0]
    # dataY2 = data[train_num + time_step:, 0]

    dataX1 = TS_X[:train_num]
    dataX2 = TS_X[train_num:]
    dataY1 = data[time_step: train_num + time_step, 0]
    dataY2 = data[train_num + time_step:, 0]

    return np.array(dataX1), np.array(dataY1), np.array(dataX2), np.array(dataY2)
def load_dataDL(dataPOV, ahead_num,N1):


    N2 = N1 + 10000

    dataAll = dataPOV[:N2, :]
    trainX, trainY, testX, testY = create_data(dataAll, N1, ahead_num)
    print("trainX", trainX.shape)
    print("trainY", trainY.shape)
    return trainX, trainY, testX, testY

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
    dataPOV = dataset.iloc[0:,0]
    dataAir = dataset.iloc[0:, 1]
    N_train = 7000
    N_test = 1000
    startNum = 1
    ahead_num = 8
    N2 = N_train + N_test
    global scaler_POV
    dataPOV = np.array(dataPOV[startNum + 1: startNum +N_train + N_test + 1]).reshape(-1, 1)
    scaler_POV = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataset_POV = scaler_POV.fit_transform(dataPOV)

    global scaler_Air
    dataAir = np.array(dataAir[startNum + 1: startNum + N_train + N_test + 1]).reshape(-1, 1)
    scaler_Air = StandardScaler(copy=True, with_mean=True, with_std=True)
    dataAir = scaler_Air.fit_transform(dataAir)
    x_trainDL, y_trainDL, x_testDL, y_testDL = load_dataDL(dataPOV, ahead_num,N_train)

    x_train, y_train, x_test, y_test = load_data_ts(N_train, N_test,  startNum, dataPOV)
    x_train1, y_train1, x_test1, y_test1 = load_data_ts(N_train, N_test, startNum, dataAir)
    print("x",y_test)
    train_x_list = [x_train.reshape(-1, ahead_num, 1),
                    x_train1.reshape(-1, ahead_num, 1)
                    ]

    test_x_list = [x_test.reshape(-1, ahead_num, 1),
                   x_test1.reshape(-1, ahead_num, 1)
                   ]
    train_y_list = [y_train, y_train1]
    test_y_list = [y_test, y_test1]

    wvlt_list, _ = load_data_wvlt(N_train, N_test, startNum, dataPOV)
    wvlt_list1, _ = load_data_wvlt(N_train, N_test, startNum, dataAir)

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

    model11 = load_model("60min.h5")
    model12 = load_model("LSTM60min.H5")
    model13 = load_model("BiLSTM60min.h5")
    model14 = load_model("CNN_LSTM60min.h5")
    model15 = load_model("CNN_BLSTM60min.h5")
    #model16 =load_model("WT_LSTM60min.h5")
   # model17 = load_model("WT_BLSTM60min.h5")
    model18=load_model("MC-WT_LSTM60min.h5")
    model19=load_model("MC-WT-BiLSTM60min.h5")


   # model_DecisionTreeRegressor = tree.DecisionTreeRegressor()  # 决策树
   # model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)  # 随机森林
   # model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=50)  # GDBT
   # model_LinearRegression = linear_model.LinearRegression()  # 线性回归
   # model_SVR = svm.SVR()  # SVR回归
  #  model_KNeighborsRegressor = neighbors.KNeighborsRegressor()  # KNN回归
  #  model_ExtraTreeRegressor = ExtraTreeRegressor()  # extra tree
    model_MLP = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=2)  # MLP
    model_BaggingRegressor = BaggingRegressor()  # bggingRegressor
  #  model_AdaboostRegressor = AdaBoostRegressor()  # adaboostRegressor

    predict_list = model11.predict(wvlt_list_test_MV)
    predict_list2=model12.predict(test_x_list )
    predict_list3=model13.predict(test_x_list )
    predict_list4=model14.predict(test_x_list )
    predict_list5=model15.predict(test_x_list )
   # predict_list6=model16.predict(wvlt_list_test_MV[0] )
   # predict_list7=model17.predict(wvlt_list_test_MV[0] )
    predict_list8=model18.predict(wvlt_list_test_MV)
    predict_list9=model19.predict(wvlt_list_test_MV)

    # predict_decideTree = pre_model(model_DecisionTreeRegressor, x_trainDL, y_trainDL, x_testDL)
    # predict_randomForest = pre_model(model_RandomForestRegressor,x_trainDL, y_trainDL, x_testDL)
    # predict_linear = pre_model(model_LinearRegression, x_trainDL, y_trainDL, x_testDL)
   # predict_svr = pre_model(model_SVR,x_trainDL, y_trainDL, x_testDL)
    # predict_kNeighbors = pre_model(model_KNeighborsRegressor, x_trainDL, y_trainDL, x_testDL)
    # predict_gradientBoosting = pre_model(model_GradientBoostingRegreessor, x_trainDL, y_trainDL, x_testDL)
    predict_bagging = pre_model(model_BaggingRegressor, x_trainDL, y_trainDL, x_testDL)
    # predict_extraTree = pre_model(model_ExtraTreeRegressor, x_trainDL, y_trainDL, x_testDL)
    predict_mlp = pre_model(model_MLP, x_trainDL, y_trainDL, x_testDL)


    predict_Pov1 = predict_list[0]
    predict_Pov1 = predict_Pov1.reshape(-1, )
    predict_Pov2 = predict_list2[0]
    predict_Pov2 = predict_Pov2.reshape(-1, )
    predict_Pov3 = predict_list3[0]
    predict_Pov3 = predict_Pov3.reshape(-1, )
    predict_Pov4 = predict_list4[0]
    predict_Pov4 = predict_Pov4.reshape(-1, )
    predict_Pov5 = predict_list5[0]
    predict_Pov5 = predict_Pov5.reshape(-1, )
  #  predict_Pov6 = predict_list6
   # predict_Pov6 = predict_Pov6.reshape(-1, )
  #  predict_Pov7 = predict_list7
   # predict_Pov7 = predict_Pov7.reshape(-1, )
    predict_Pov8 = predict_list8[0]
    predict_Pov8 = predict_Pov8.reshape(-1, )
    predict_Pov9 = predict_list9[0]
    predict_Pov9 = predict_Pov9.reshape(-1, )
    test_y_pov = test_y_list[0]

#    global scaler_POV
   # predict_decideTree = scaler_POV.inverse_transform(predict_decideTree)
   # predict_randomForest = scaler_POV.inverse_transform(predict_randomForest)
   # predict_linear = scaler_POV.inverse_transform(predict_linear)
   # predict_svr = scaler_POV.inverse_transform(predict_svr)
   # predict_kNeighbors =scaler_POV.inverse_transform(predict_kNeighbors)
    #predict_gradientBoosting = scaler_POV.inverse_transform(predict_gradientBoosting)
   # predict_extraTree = scaler_POV.inverse_transform(predict_extraTree)
    #predict_mlp = scaler_POV.inverse_transform(predict_mlp)
   # predict_bagging = scaler_POV.inverse_transform(predict_bagging)
   # predict_adaboost = scaler_POV.inverse_transform(predict_adaboost)
    dataY = scaler_POV.inverse_transform(y_testDL)
    # mae_decideTree = MAE1(dataY, predict_decideTree)
    # rmse_decideTree = RMSE1(dataY, predict_decideTree)
    # mape_decideTree = MAPE1(dataY, predict_decideTree)
    # r2_decideTree = r2_score(dataY, predict_decideTree)

    # print("======================================================")
    # print("rmse_decideTree:", rmse_decideTree)
    # print("mape_decideTree:", mape_decideTree)
    # print("mae_decideTree:", mae_decideTree)
    # print("R_decideTree", r2_decideTree)
    # rmse_randomForest = RMSE1(dataY, predict_randomForest)
    # mape_randomForest = MAPE1(dataY, predict_randomForest)
    # mae_randomForest = MAE1(dataY, predict_randomForest)
    # r2_randomForest = r2_score(dataY, predict_randomForest)
    #
    # print("R2_randomForest", r2_randomForest)
    # adjust_R = 1 - ((1 - r2_randomForest) * (1000 - 1)) / (1000 - 1 - 1)
    # print("adjust_R", adjust_R)
    # print("mae_randomForest:", mae_randomForest)
    # print("rmse_randomForest:", rmse_randomForest)
    # print("mape_randomForest:", mape_randomForest)
    # print("r2_randomForest", r2_randomForest)
    #
    # rmse_linear = RMSE1(dataY, predict_linear)
    # mape_linear = MAPE1(dataY, predict_linear)
    # mae_linear = MAE1(dataY, predict_linear)
    # r2_linear = r2_score(dataY, predict_linear)
    # print("R2_linear", r2_linear)
    # print("mae_linear:", mae_linear)
    # print("rmse_linear:", rmse_linear)
    # print("mape_linear:", mape_linear)
    #
    # rmse_svr = RMSE1(dataY, predict_svr)
    # mape_svr = MAPE1(dataY, predict_svr)
    # mae_svr = MAE1(dataY, predict_svr)
    # r2_svr = r2_score(dataY, predict_svr)
    # print("R2_svr", r2_svr)
    # print("mae_svr:", mae_svr)
    # print("rmse_svr:", rmse_svr)
    # print("mape_svr:", mape_svr)
    #
    # rmse_kNeighbors = RMSE1(dataY, predict_kNeighbors)
    # mape_kNeighbors = MAPE1(dataY, predict_kNeighbors)
    # mae_kNeighbors = MAE1(dataY, predict_kNeighbors)
    # r2_kNeighbors = r2_score(dataY, predict_kNeighbors)
    # print("R2_kNeighbors", r2_kNeighbors)
    # print("mae_kNeighbors:", mae_kNeighbors)
    # print("rmse_kNeighbors:", rmse_kNeighbors)
    # print("mape_kNeighbors:", mape_kNeighbors)
    #
    # rmse_mlp = RMSE1(dataY, predict_mlp)
    # mape_mlp = MAPE1(dataY, predict_mlp)
    # mae_mlp = MAE1(dataY, predict_mlp)
    # r2_mlp = r2_score(dataY, predict_mlp)
    # print("R2_mlp", r2_mlp)
    # print("mae_mlp:", mae_mlp)
    # print("rmse_mlp:", rmse_mlp)
    # print("mape_mlp:", mape_mlp)
    #
    # rmse_gradientBoosting = RMSE1(dataY, predict_gradientBoosting)
    # mape_gradientBoosting = MAPE1(dataY, predict_gradientBoosting)
    # mae_gradientBoosting = MAE1(dataY, predict_gradientBoosting)
    # r2_gradientBoosting = r2_score(dataY, predict_gradientBoosting)
    # print("R2_gradientBoosting ", r2_gradientBoosting)
    # print("mae_gradientBoosting:", mae_gradientBoosting)
    # print("rmse_gradientBoosting:", rmse_gradientBoosting)
    # print("mape_gradientBoosting:", mape_gradientBoosting)
    #
    # rmse_extraTree = RMSE1(dataY, predict_extraTree)
    # mape_extraTree = MAPE1(dataY, predict_extraTree)
    # mae_extraTree = MAE1(dataY, predict_extraTree)
    # r2_extraTree = r2_score(dataY, predict_extraTree)
    # print("R2_extraTree ", r2_extraTree)
    # print("mae_extraTree:", mae_extraTree)
    # print("rmse_extraTree:", rmse_extraTree)
    # print("mape_extraTree:", mape_extraTree)
    #
    # rmse_bagging = RMSE1(dataY, predict_bagging)
    # mape_bagging = MAPE1(dataY, predict_bagging)
    # mae_bagging = MAE1(dataY, predict_bagging)
    # r2_bagging = r2_score(dataY, predict_bagging)
    # print("R2_bagging ", r2_bagging)
    # print("mae_bagging:", mae_bagging)
    # print("rmse_bagging:", rmse_bagging)
    # print("mape_bagging:", mape_bagging)
    #
    # rmse_adaboost = RMSE1(dataY, predict_adaboost)
    # mape_adaboost = MAPE1(dataY, predict_adaboost)
    # mae_adaboost = MAE1(dataY, predict_adaboost)
    # r2_adaboost = r2_score(dataY, predict_adaboost)
    # print("R2_adaboost ", r2_adaboost)
    # print("mae_adaboost:", mae_adaboost)
    # print("rmse_bagging:", rmse_adaboost)
    # print("mape_adaboost:", mape_adaboost)
    # print("zhenshi:",test_y_pov)
    forecast_skill = skill(test_y_pov, predict_Pov1)

    print("skill", forecast_skill)
    mbe = MBE(test_y_pov, predict_Pov1)
    print(mbe)
    smape1=smape(test_y_pov, predict_Pov1)
    print("mymodel:",smape1)
    rmse_WN = RMSE1(test_y_pov, predict_Pov1)
    print("rmse", rmse_WN)
    mape_WN = MAPE1(test_y_pov, predict_Pov1)
    print("mape", mape_WN)
    mae_WN = MAE1(test_y_pov, predict_Pov1)
    print("mae", mae_WN)
    mse_WN = MSE1(test_y_pov, predict_Pov1)
    print("mse", mse_WN)
    r2_wn = r2_score(test_y_pov, predict_Pov1)
    print("R2", r2_wn)

    print("skill", forecast_skill)
    mbe2 = MBE(test_y_pov, predict_Pov2)
    print(mbe2)
    smape2 = smape(test_y_pov, predict_Pov2)
    print("duibi:", smape2)
    rmse_WN2 = RMSE1(test_y_pov, predict_Pov2)
    print("rmse2", rmse_WN2)
    mape_WN2 = MAPE1(test_y_pov, predict_Pov2)
    print("mape2", mape_WN2)
    mae_WN2 = MAE1(test_y_pov, predict_Pov2)
    print("mae2", mae_WN2)
    mse_WN2 = MSE1(test_y_pov, predict_Pov2)
    print("mse2", mse_WN2)
    r2_wn2 = r2_score(test_y_pov, predict_Pov2)
    print("R2_2", r2_wn2)

    print("skill", forecast_skill)
    mbe3 = MBE(test_y_pov, predict_Pov3)
    print(mbe3)
    smape3 = smape(test_y_pov, predict_Pov3)
    print("duibi3:", smape3)
    rmse_WN3 = RMSE1(test_y_pov, predict_Pov3)
    print("rmse3", rmse_WN3)
    mape_WN3 = MAPE1(test_y_pov, predict_Pov3)
    print("mape3", mape_WN3)
    mae_WN3 = MAE1(test_y_pov, predict_Pov3)
    print("mae3", mae_WN3)
    mse_WN3 = MSE1(test_y_pov, predict_Pov3)
    print("mse3", mse_WN3)
    r2_wn3 = r2_score(test_y_pov, predict_Pov3)
    print("R2_3", r2_wn3)

    print("skill", forecast_skill)
    mbe4 = MBE(test_y_pov, predict_Pov4)
    print(mbe4)
    smape4 = smape(test_y_pov, predict_Pov4)
    print("duibi4:", smape4)
    rmse_WN4 = RMSE1(test_y_pov, predict_Pov4)
    print("rmse4", rmse_WN4)
    mape_WN4 = MAPE1(test_y_pov, predict_Pov4)
    print("mape4", mape_WN4)
    mae_WN4 = MAE1(test_y_pov, predict_Pov4)
    print("mae4", mae_WN4)
    mse_WN4 = MSE1(test_y_pov, predict_Pov4)
    print("mse4", mse_WN4)
    r2_wn4= r2_score(test_y_pov, predict_Pov4)
    print("R2_4", r2_wn4)

    print("skill", forecast_skill)
    mbe5 = MBE(test_y_pov, predict_Pov5)
    print(mbe5)
    smape5 = smape(test_y_pov, predict_Pov5)
    print("duibi5:", smape5)
    rmse_WN5 = RMSE1(test_y_pov, predict_Pov5)
    print("rmse5", rmse_WN5)
    mape_WN5 = MAPE1(test_y_pov, predict_Pov5)
    print("mape5", mape_WN5)
    mae_WN5 = MAE1(test_y_pov, predict_Pov5)
    print("mae5", mae_WN5)
    mse_WN5 = MSE1(test_y_pov, predict_Pov5)
    print("mse5", mse_WN5)
    r2_wn5 = r2_score(test_y_pov, predict_Pov5)
    print("R2_5", r2_wn5)

    print("skill", forecast_skill)
    mbe8 = MBE(test_y_pov, predict_Pov8)
    print(mbe8)
    smape8 = smape(test_y_pov, predict_Pov8)
    print("duibi8:", smape8)
    rmse_WN8= RMSE1(test_y_pov, predict_Pov8)
    print("rmse8", rmse_WN8)
    mape_WN8 = MAPE1(test_y_pov, predict_Pov8)
    print("mape8", mape_WN8)
    mae_WN8 = MAE1(test_y_pov, predict_Pov8)
    print("mae8", mae_WN8)
    mse_WN8 = MSE1(test_y_pov, predict_Pov8)
    print("mse8", mse_WN8)
    r2_wn8 = r2_score(test_y_pov, predict_Pov8)
    print("R2_8", r2_wn8)

    print("skill", forecast_skill)
    mbe9 = MBE(test_y_pov, predict_Pov9)
    print(mbe9)
    smape9 = smape(test_y_pov, predict_Pov9)
    print("duibi9:", smape9)
    rmse_WN9 = RMSE1(test_y_pov, predict_Pov9)
    print("rmse9", rmse_WN9)
    mape_WN9 = MAPE1(test_y_pov, predict_Pov9)
    print("mape9", mape_WN9)
    mae_WN9 = MAE1(test_y_pov, predict_Pov9)
    print("mae9", mae_WN9)
    mse_WN9 = MSE1(test_y_pov, predict_Pov9)
    print("mse9", mse_WN9)
    r2_wn9 = r2_score(test_y_pov, predict_Pov9)
    print("R2_9", r2_wn9)
    dataframe = pd.DataFrame({"True":test_y_pov, "Proposed":predict_Pov1, "bagging":predict_bagging,"mlp":predict_mlp,"LSTM":predict_Pov2,"BLSTM":predict_Pov3,"CNN-LSTM":predict_Pov4,"CNN-BLSTM":predict_Pov5,"WT-LSTM":predict_Pov8,"WT-BLSTM":predict_Pov9})

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r"60minnew.csv", sep=',')



    plt.figure(1, figsize=(15, 5))

    plt.plot(test_y_pov[:], "black", label="True",linewidth=3, linestyle='--', marker='.')
    plt.plot(predict_bagging[:], 'thistle',label="Bagging", linewidth=1.25)
    plt.plot(predict_mlp[:], 'beige',label="MLP", linewidth=1.25)
    plt.plot(predict_Pov2[:], 'peru',label="LSTM", linewidth=1.25)
    plt.plot(predict_Pov3[:], 'lightgreen',label="BiLSTM", linewidth=1.25)
    plt.plot(predict_Pov4[:], 'lightskyblue',label="CNN-LSTM", linewidth=1.25)
    plt.plot(predict_Pov5[:], 'honeydew',label="CNN-BiLSTM", linewidth=1.25)
   # plt.plot(predict_Pov6[:], 'violet', label='WT-LSTM', linewidth=1.5)
   # plt.plot(predict_Pov7[:], 'fuchsia', label='WT-BiLSTM', linewidth=1.5)
    plt.plot(predict_Pov8[:], 'aqua',label='WT-LSTM', linewidth=1.25)
    plt.plot(predict_Pov9[:], 'khaki',label='WT-BiLSTM', linewidth=1.25)

    plt.plot(predict_Pov1[:], 'r',label='Proposed', linewidth=1.75,marker='.')
    legend_font = {"family": "Times New Roman",
                   "size": 14}
    plt.legend(loc='upper right',prop=legend_font)

    plt.xlabel("Time interval (60min)", fontsize=14)
    plt.ylabel("GHI(W/m2)", fontsize=14)

    plt.yticks(fontproperties='Times New Roman', fontsize=14)
    plt.xticks(fontproperties='Times New Roman', fontsize=14)
   # plt.title('Global horizontal irradiance forecast results (60min)',fontproperties='Times New Roman',fontsize=16)

    plt.grid()
    plt.show()
if __name__ == "__main__":
    filename="C:\\Users\\Administrator\\PycharmProjects\\solar\\GHI_irradiance_60min.csv"
    ahead_num = 8
    load_data(filename,ahead_num)
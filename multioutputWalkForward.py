import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
# import to hide the annoying warnings about numba.jit when importing shap library
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras import layers, metrics, Sequential, models, regularizers
from keras_tuner.tuners import RandomSearch
from sklearn.linear_model import Lasso
import shutil, os
import pickle
from functools import partial
import argparse
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from yellowbrick.regressor.residuals import residuals_plot
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import VotingRegressor
from generateMetricPlots import mainPlotting
from sklearn.multioutput import RegressorChain

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, window, testWindow): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    econData = readEconData("Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    trainDf, testDf = econData.iloc[:window], econData.iloc[window:window+testWindow]
    xTrain, xTest = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"]
    # For each row in the xTrain and xTest dataframes, we want to add the next row of x's to its corresponding y's
    # For example, the second row of X's is added to the first row of y's, the third row of X's is added to the second row of y's, etc.
    # This is done to create a multioutput regression problem
    yTrain, yTest = trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
    neededXs = econData.iloc[1:window+testWindow+1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
    # Reset the indices of neededXs
    neededXs.reset_index(drop=True, inplace=True)
    # Add the X columns to Y columns
    yTrain = pd.concat([yTrain, neededXs.iloc[:window]], axis=1)
    yTest = pd.concat([yTest, neededXs.iloc[window:window+testWindow]], axis=1)
    return xTrain, yTrain, xTest, yTest

def plotPredictions(xTest, yTest, model, modelName):
    # Plot the predictions of the model on the xTest data
    predictions = model.predict(xTest)
    plt.plot(predictions, label="Predictions")
    plt.plot(yTest, label="Actual", color="orange")
    plt.legend()
    plt.title("Predictions vs Actual for "+modelName)
    plt.gcf().canvas.manager.set_window_title("Predictions vs Actual for "+modelName)
    plt.show()

def getModelMetrics(x, y, model, modelName, training=True):
    # Get the R^2, Adjusted R^2, MSE, RMSE, MAE, and Pearson's Correlation Coefficent for the model
    # check if y is a numpy.ndarray
    # Drop all columns in Y except for the AnnualizedMoM-CPI-Inflation column
    y.drop(y.columns.difference(["AnnualizedMoM-CPI-Inflation"]), axis=1, inplace=True)
    if training==False and not(y.__class__ == np.ndarray):
        y = np.array([value[0] for value in y.values.tolist()])
    elif training==False:
        y = np.array([value[0] for value in y.tolist()])
    predictions = model.predict(x)
    # Drop all the predictions exceprt for the first column
    predictions = predictions[:, 0]
    r2 = r2_score(y, predictions).round(3)
    adjR2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
    adjR2 = adjR2.round(3)
    mse = mean_squared_error(y, predictions).round(3)
    rmse = np.sqrt(mse).round(3)
    predictions = predictions.reshape(predictions.size, 1)
    mae = np.mean(np.abs(predictions - y)).round(3)
    corr = np.corrcoef(predictions.T, y.T)[0,1].round(3)
    if training:
        print("Training Metrics for "+modelName+":")
        print("R^2: "+str(r2))
        print("Adjusted R^2: "+str(adjR2))
    else:
        print("Testing Metrics for "+modelName+":")
    print("MSE: "+str(mse))
    print("RMSE: "+str(rmse))
    print("MAE: "+str(mae))
    print("Pearson's Correlation Coefficient: "+str(corr))
    # if not training:
    #     plotPredictions(x, y, model, modelName) 
    # if modelName ==  "LR" and training:
    #     plotLRResiduals(x, y.values.tolist(), model)
    #     print("Displayed LR residuals plot for training data")
    #     getShapPlot(x, model, modelName)
    #     print("Displayed LR SHAP plot for training data")
    # elif modelName == "LR" and not training:
    #     plotLRResiduals(x, y.tolist(), model)
    #     print("Displayed LR residuals plot for testing data")
    # elif modelName == "RF" and training:
    #     getShapPlot(x, model, modelName)
    #     print("Displayed RF SHAP plot for training data")
    return r2, adjR2, mse, rmse, mae, corr

def compileMetrics(metricsDict):
    # compile the metrics into two dataframes, the first for training metrics and the second for testing metrics
    trainingMetrics = pd.DataFrame(columns = ["Train R^2", "Train Adjusted R^2", "Train MSE", "Train RMSE", "Train MAE", "Train Pearson's Correlation Coefficient"])
    testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
    for key in metricsDict.keys():
            trainingMetrics.loc[key] = metricsDict[key][:6]
            testingMetrics.loc[key] = metricsDict[key][6:]
    return trainingMetrics, testingMetrics

def getSampleWeights(numSamples, testWindow):
    c = 50
    if numSamples >= 200:
        weightsList = [i for i in range(1, numSamples+1)]
    else:
        weightsList = [i for i in range(1, testWindow+1)]
    s = sum(weightsList)
    sampleWeights = [w * (c/s) for w in weightsList]
    return sampleWeights

def trainEvalLR(window, testWindow, loadModel=False):
    xTrain, yTrain, xTest, yTest = makeTrainTest("LR", window, testWindow)
    sampleWeights = getSampleWeights(window, testWindow)
    if loadModel:
        myLR = pickle.load(open("MultioutputModels/LRModel.pickle", "rb"))
        print("LR Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for loaded LR")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        myLR = RegressorChain(base_estimator=LinearRegression())
        myLR.fit(xTrain, yTrain, sample_weight=sampleWeights)
        print("Finished training LR")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        print("Finished displaying training metrics for LR")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for newly-trained LR")
        pickle.dump(myLR, open("MultioutputModels/LRModel.pickle", "wb"))
        print("LR Model Saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def plotLRResiduals(xTrain, yTrain, LR):
    print("X length: " + str(len(xTrain)))
    print("Y length: " + str(len(yTrain)))
    if isinstance(yTrain[0], float):
        yTrain = [[val] for val in yTrain]
    residPlot = residuals_plot(LR, X_train=xTrain, y_train = yTrain, is_fitted=True)

def trainEvalRF(window, testWindow, loadModel=False):
    xTrain, yTrain, xTest, yTest = makeTrainTest("RF", window, testWindow)
    sampleWeights = getSampleWeights(window, testWindow)
    if loadModel:
        myRF = pickle.load(open("MultioutputModels/RFModel.pickle", "rb"))
        print("RF Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for loaded RF")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        myRF = RegressorChain(base_estimator=RandomForestRegressor())
        myRF.fit(xTrain, yTrain, sample_weight=sampleWeights)
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for newly-trained RF")
        pickle.dump(myRF, open("MultioutputModels/RFModel.pickle", "wb"))
        print("RF Model Saved")
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    loadModel = False
    pre2020trainWindowSize, pre2020testWindowSize = [200 + 10*i for i in range(15)], 10
    pre2020trainMetrics, pre2020testMetrics = [], []
    post2020trainWindowSize, post2020testWindowSize = [342 + 2*i for i in range(18)], 2
    post2020trainMetrics, post2020testMetrics = [], []
    for count, window in enumerate(pre2020trainWindowSize):
        if count == 0:
            metricsDict = {"LR"+str(window): trainEvalLR(window,pre2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window,pre2020testWindowSize, loadModel)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
        else:
            metricsDict = {"LR"+str(window): trainEvalLR(window,pre2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window,pre2020testWindowSize, loadModel)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
    for count, window in enumerate(post2020trainWindowSize):
        metricsDict = {"LR"+str(window): trainEvalLR(window,pre2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window,pre2020testWindowSize, loadModel)}
        train, test = compileMetrics(metricsDict)
        post2020trainMetrics.append(train)
        post2020testMetrics.append(test)
    
    pre2020train, pre2020test = pd.concat(pre2020trainMetrics), pd.concat(pre2020testMetrics)
    post2020train, post2020test = pd.concat(post2020trainMetrics), pd.concat(post2020testMetrics)
    # add a row at the end which is the average of each column for rows containing "LR"
    pre2020train.loc['Pre2020LRavg'] = pre2020train.loc[pre2020train.index.str.contains("LR")].mean()
    pre2020test.loc['Pre2020LRavg'] = pre2020test.loc[pre2020test.index.str.contains("LR")].mean()
    post2020train.loc['Post2020LRavg'] = post2020train.loc[post2020train.index.str.contains("LR")].mean()
    post2020test.loc['Post2020LRavg'] = post2020test.loc[post2020test.index.str.contains("LR")].mean()
    # add a row at the end which is the average of each column for rows containing "RF"
    pre2020train.loc['Pre2020RFavg'] = pre2020train.loc[pre2020train.index.str.contains("RF")].mean()
    pre2020test.loc['Pre2020RFavg'] = pre2020test.loc[pre2020test.index.str.contains("RF")].mean()
    post2020train.loc['Post2020RFavg'] = post2020train.loc[post2020train.index.str.contains("RF")].mean()
    post2020test.loc['Post2020RFavg'] = post2020test.loc[post2020test.index.str.contains("RF")].mean()
    # # add a row at the end which is the average of each column for rows containing "NN"
    # pre2020train.loc['Pre2020NNavg'] = pre2020train.loc[pre2020train.index.str.contains("NN")].mean()
    # pre2020test.loc['Pre2020NNavg'] = pre2020test.loc[pre2020test.index.str.contains("NN")].mean()
    # post2020train.loc['Post2020NNavg'] = post2020train.loc[post2020train.index.str.contains("NN")].mean()
    # post2020test.loc['Post2020NNavg'] = post2020test.loc[post2020test.index.str.contains("NN")].mean()
    # # add a row at the end which is the average of each column for rows containing "RNN"
    # pre2020train.loc['Pre2020RNNavg'] = pre2020train.loc[pre2020train.index.str.contains("RNN")].mean()
    # pre2020test.loc['Pre2020RNNavg'] = pre2020test.loc[pre2020test.index.str.contains("RNN")].mean()
    # post2020train.loc['Post2020RNNavg'] = post2020train.loc[post2020train.index.str.contains("RNN")].mean()
    # post2020test.loc['Post2020RNNavg'] = post2020test.loc[post2020test.index.str.contains("RNN")].mean()
    # # add a row at the end which is the average of each column for rows containing "Ensemble"
    # pre2020train.loc['Pre2020Ensembleavg'] = pre2020train.loc[pre2020train.index.str.contains("Ensemble")].mean()
    # pre2020test.loc['Pre2020Ensembleavg'] = pre2020test.loc[pre2020test.index.str.contains("Ensemble")].mean()
    # post2020train.loc['Post2020Ensembleavg'] = post2020train.loc[post2020train.index.str.contains("Ensemble")].mean()
    # post2020test.loc['Post2020Ensembleavg'] = post2020test.loc[post2020test.index.str.contains("Ensemble")].mean()
    # combine the train and test dataframes into one
    train = pd.concat([pre2020train, post2020train])
    test = pd.concat([pre2020test, post2020test])
    # move rows that contain "avg" to the bottom
    train = train[~train.index.str.contains("avg")].append(train[train.index.str.contains("avg")])
    test = test[~test.index.str.contains("avg")].append(test[test.index.str.contains("avg")])
    # START OF TOTAL AVERAGE CODE
    temp = train.loc[train.index.str.contains("Pre2020LRavg")]*10 
    temp2 = train.loc[train.index.str.contains("Post2020LRavg")]*2
    # Add each value of dataframe temp2 to temp's values, leading to temp's values array being of shape (1, 9)
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    # Divide temp's values by 12
    temp = temp/12
    # Rename the index of temp to TotalLRAvg
    temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
    # Add temp as a row to the end of the train dataframe
    train = pd.concat([train, temp])
    # Rename the last row in train to TotalLRAvg
    # train.rename(index={train.index[-1]:'TotalLRAvg'}, inplace=True)
    # Repeat the above steps for the test dataframe
    temp = test.loc[test.index.str.contains("Pre2020LRavg")]*10
    temp2 = test.loc[test.index.str.contains("Post2020LRavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalLRAvg
    temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
    test = pd.concat([test, temp])
    test.rename(index={test.index[-1]:'TotalLRAvg'}, inplace=True)
    # Repeat the above steps for the RF model
    temp = train.loc[train.index.str.contains("Pre2020RFavg")]*10
    temp2 = train.loc[train.index.str.contains("Post2020RFavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalRFAvg
    temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRFAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020RFavg")]*10
    temp2 = test.loc[test.index.str.contains("Post2020RFavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalRFAvg
    temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalRFAvg'}, inplace=True)
    '''
    # Repeat the above steps for the NN model
    temp = train.loc[train.index.str.contains("Pre2020NNavg")]*10
    temp2 = train.loc[train.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020NNavg")]*10
    temp2 = test.loc[test.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalNNAvg'}, inplace=True)
    # Repeat the above steps for the RNN model
    temp = train.loc[train.index.str.contains("Pre2020RNNavg")]*10
    temp2 = train.loc[train.index.str.contains("Post2020RNNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020RNNavg")]*10
    temp2 = test.loc[test.index.str.contains("Post2020RNNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalRNNAvg'}, inplace=True)
    # Repeat the above steps for the Ensemble model
    temp = train.loc[train.index.str.contains("Pre2020Ensembleavg")]*10
    temp2 = train.loc[train.index.str.contains("Post2020Ensembleavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalEnsembleAvg
    temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalEnsembleAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020Ensembleavg")]*10
    temp2 = test.loc[test.index.str.contains("Post2020Ensembleavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/12
    # Rename the index of temp to TotalEnsembleAvg
    temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalEnsembleAvg'}, inplace=True)
    '''
    # move the rows that contain "Avg" to the bottom
    train = train[~train.index.str.contains("Avg")].append(train[train.index.str.contains("Avg")])
    test = test[~test.index.str.contains("Avg")].append(test[test.index.str.contains("Avg")])
    # save the dataframes to excel files
    train.to_excel("MultioutputMetrics/rollingTrainMetrics.xlsx")
    test.to_excel("MultioutputMetrics/rollingTestMetrics.xlsx")
    print("Program Done")
    mainPlotting()

if __name__ == "__main__":
    main()

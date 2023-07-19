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
from datetime import datetime

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, start, end): # Make X, Y for predictions
    # Read econ data
    econData = readEconData("Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    if modelName != "RNN":
        # Filter out the data to only include the time period we want
        econData = econData[(econData["Date"] >= start) & (econData["Date"] <= end)]
        # drop the date column
        econData.drop("Date", axis=1, inplace=True)
        # scale the data using StandardScaler
        scaler = StandardScaler()
        econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        x, y = econData.loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"], econData.loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        return x, y
    else:
        # Filter out the data to be from start - 12 months to end
        # subtract 12 months from start
        start = start - pd.DateOffset(months=12)
        econData = econData[(econData["Date"] >= start) & (econData["Date"] <= end)]
        # drop the date column
        econData.drop("Date", axis=1, inplace=True)
        # scale the data using StandardScaler
        scaler = StandardScaler()
        econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        # split the data into x and y
        x, y = econData.loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"], econData.loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        return x, y

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
    if training==False and not(y.__class__ == np.ndarray):
        y = np.array([value[0] for value in y.values.tolist()])
    elif training==False:
        y = np.array([value[0] for value in y.tolist()])
    predictions = model.predict(x)
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
    if not training:
        plotPredictions(x, y, model, modelName) 
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

def getShapPlot(x, model, modelName):
    # reference: https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    if modelName == "LR":
        explainer = shap.explainers.Linear(model, x)
        shap_values = explainer.shap_values(x)
        shap.summary_plot(shap_values, x, feature_names=x.columns, plot_type="bar", show=False)
        plt.title("Feature Importance for Linear Regression")
        plt.gcf().canvas.manager.set_window_title("Feature Importance for Linear Regression")
        plt.gcf().set_size_inches(10,6)
        plt.show()
    elif modelName == "RF":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        shap.summary_plot(shap_values, x, feature_names=x.columns, plot_type="bar", show=False)
        plt.title("Feature Importance for Random Forest")
        plt.gcf().canvas.manager.set_window_title("Feature Importance for Random Forest")
        plt.gcf().set_size_inches(10,6)
        plt.show()
    else:
        print("Shap Plot Error: Unsupported model to get SHAP Plot")

def compileMetrics(metricsDict, start, end, loadModel=True):
    # compile the metrics into two dataframes, the first for training metrics and the second for testing metrics
    if loadModel:
        testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
        for key in metricsDict.keys():
            testingMetrics.loc[key] = metricsDict[key]
    startString = str(start).replace(":", "-").replace("00-00-00","")
    endString = str(end).replace(":", "-").replace("00-00-00","")
    testingMetrics.to_excel("Metrics/TestingMetrics"+startString+" to "+endString+".xlsx")
    print("Metrics saved to Excel files")
    return testingMetrics

def trainEvalLR(start, end, loadModel=True):
    xTest, yTest = makeTrainTest("LR", start, end)
    if loadModel:
        myLR = pickle.load(open("Models/LRModel.pickle", "rb"))
        print("LR Model Loaded")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for loaded LR")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def plotLRResiduals(xTrain, yTrain, LR):
    print("X length: " + str(len(xTrain)))
    print("Y length: " + str(len(yTrain)))
    if isinstance(yTrain[0], float):
        yTrain = [[val] for val in yTrain]
    residPlot = residuals_plot(LR, X_train=xTrain, y_train = yTrain, is_fitted=True)

def trainEvalRF(start, end, loadModel=True):
    xTest, yTest = makeTrainTest("RF", start, end)
    if loadModel:
        myRF = pickle.load(open("Models/RFModel.pickle", "rb"))
        print("RF Model Loaded")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for loaded RF")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalNN(start, end, loadModel=True):
    xTest, yTest = makeTrainTest("NN", start, end)
    if loadModel:
        myNN = keras.models.load_model("Models/NNModel.h5")
        print("Loaded NN")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myNN, "NN", training=False)
        print(myNN.summary())
        print("Finished displaying testing metrics for loaded NN")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalRNN(start, end, loadModel=True):
    xTest, yTest = makeTrainTest("RNN", start, end)
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
    timestep = 12 # number of timesteps to look back
    # Split the xTrain and xTeset into rolling windows of size timestep, and have yTrain and yTest be the next value
    rnnXTest = np.array([xTest[i:i+timestep] for i in range(len(xTest)-timestep)])
    rnnYTest = np.array([yTest.values[i+timestep] for i in range(len(yTest)-timestep)])
    rnnXTest.reshape(len(rnnXTest), timestep, 23)
    if loadModel:
        myRNN = keras.models.load_model("Models/RNNModel.h5")
        print("Loaded RNN")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for loaded RNN")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalEnsemble(start, end, loadModel=True):
    xTest, yTest = makeTrainTest("Ensemble", start, end)
    # Load in the individual models
    if loadModel:
        myLR = pickle.load(open("Models/EnsembleModel.pickle", "rb"))
        print("Ensemble Model Loaded")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "Ensemble", training=False)
        print("Finished displaying testing metrics for loaded Ensemble")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Starting month year in format month_year, both as numbers",  required=True)
    parser.add_argument("--end", help="Ending month year in format month_year, both as numbers", required=True)
    args = parser.parse_args()
    startDateTime = datetime.strptime(args.start, "%m_%Y")
    endDateTime = datetime.strptime(args.end, "%m_%Y")
    # startDateTime = datetime.strptime("1_2020", "%m_%Y")
    # endDateTime = datetime.strptime("3_2023", "%m_%Y")
    # set each date's time to be 12:00:00 AM
    startDateTime = startDateTime.replace(hour=0, minute=0, second=0, microsecond=0)
    endDateTime = endDateTime.replace(hour=0, minute=0, second=0, microsecond=0)
    print("Start date: "+str(startDateTime))
    print("End date: "+str(endDateTime))
    metricsDict = {"LR "+str(startDateTime)+" to "+str(endDateTime): trainEvalLR(startDateTime, endDateTime), "RF"+str(startDateTime)+" to "+str(endDateTime): trainEvalRF(startDateTime, endDateTime), "NN"+str(startDateTime)+" to "+str(endDateTime): trainEvalNN(startDateTime, endDateTime), "RNN"+str(startDateTime)+" to "+str(endDateTime): trainEvalRNN(startDateTime, endDateTime) ,"Ensemble"+str(startDateTime)+" to "+str(endDateTime): trainEvalEnsemble(startDateTime, endDateTime)}
    compileMetrics(metricsDict, startDateTime, endDateTime)
    print('Program done')

if __name__ == "__main__":
    main()
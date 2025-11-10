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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

"""
Walk-forward validation for inflation forecasting models (INTERPOLATED DATA VERSION).

This script trains and evaluates multiple models (Linear Regression, Random Forest, ARIMA, 
Neural Network, RNN, and Ensemble) across different time periods (pre-2020, transition, post-2020)
using interpolated monthly data.

Usage:
    # Train all models (default)
    python walkForwardPrePost2020INTERP.py
    
    # Train specific models
    python walkForwardPrePost2020INTERP.py --models LR RF ARIMA
    
    # Train only neural networks
    python walkForwardPrePost2020INTERP.py --models NN RNN
    
    # Load existing models instead of training from scratch
    python walkForwardPrePost2020INTERP.py --load-model
    
    # Combine options
    python walkForwardPrePost2020INTERP.py --models LR ARIMA NN --load-model

Available models: LR, RF, ARIMA, NN, RNN, Ensemble, all
"""

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, window, testWindow, secondTime=False): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    econData = readEconData("Data/ConstructedDataframes/INTERPAllEcon1990AndCOVIDWithLags.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    if modelName != "RNN":
        if not secondTime:
            # split into train/test 
            trainDf, testDf = econData.iloc[:window], econData.iloc[window:window+testWindow]
            # split into x and y
            xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
            xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
        else:
            if window == 346: #First post2020 
                trainDf, testDf = econData.iloc[window-6:window], econData.iloc[window:window+testWindow]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
            else:
                trainDf, testDf = econData.iloc[window-testWindow:window], econData.iloc[window:window+testWindow]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
    else:
        # Specific train/test split for the RNN due to it needing to pull some values in the train window for the xTest
        if not secondTime:
            trainDf = econData.iloc[:window]
            xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
            xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
            yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        else:
            if window == 346: #First post2020
                trainDf = econData.iloc[window-6-12:window]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
                yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
            else:
                trainDf = econData.iloc[window-testWindow-12:window]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
                yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
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
    if training==False and not(y.__class__ == np.ndarray):
        y = np.array([value[0] for value in y.values.tolist()])
    elif training==False:
        y = np.array([value[0] for value in y.tolist()])
    """Reduce the x and y to be every 4th value to remove the synthetic data"""
    # if x is a dataframe 
    if x.__class__ == pd.core.frame.DataFrame:
        x = x.iloc[::4, :]
    # if x is a series
    elif x.__class__ == pd.core.series.Series:
        x = x.iloc[::4]
    # otherwise if it's just a numpy array
    else:
        x = x[::4]
    # if y is a dataframe convert it to a numpy array and then reduce it to be every 4th value
    if y.__class__ == pd.core.frame.DataFrame:
        y = np.array([value[0] for value in y.values.tolist()])
        y = y[::4]
    # if y is a series convert it to a numpy array and then reduce it to be every 4th value
    elif y.__class__ == pd.core.series.Series:
        y = np.array([value[0] for value in y.tolist()])
        y = y[::4]
    # otherwise if it's just a numpy array reduce it to be every 4th value
    else:
        y = y[::4]
    predictions = model.predict(x)
    r2 = round(r2_score(y, predictions),3)
    adjR2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
    adjR2 = round(adjR2,3)
    mse = round(mean_squared_error(y, predictions),3)
    rmse = round(np.sqrt(mse),3)
    predictions = predictions.reshape(predictions.size, 1)
    mae = round(np.mean(np.abs(predictions - y)),3)
    corr = round(np.corrcoef(predictions.T, y.T)[0,1], 3)
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

def compileMetrics(metricsDict):
    # compile the metrics into two dataframes, the first for training metrics and the second for testing metrics
    trainingMetrics = pd.DataFrame(columns = ["cvMSE", "cvRMSE", "cvMAE","Train R^2", "Train Adjusted R^2", "Train MSE", "Train RMSE", "Train MAE", "Train Pearson's Correlation Coefficient"])
    testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
    for key in metricsDict.keys():
        if "LR" in key or "RF" in key or "ARIMA" in key:
            trainingMetrics.loc[key] = metricsDict[key][:9]
            testingMetrics.loc[key] = metricsDict[key][9:]
        else:
            trainingMetrics.loc[key] = [0,0,0]+list(metricsDict[key][:6])
            testingMetrics.loc[key] = list(metricsDict[key][6:])
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
        myLR = pickle.load(open("InterpModels/LRModel.pickle", "rb"))
        print("LR Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for loaded LR")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        myLR = LinearRegression()
        # Perform cross validation
        print("Performing cross validation for LR")
        cvScores = cross_validate(myLR, xTrain, yTrain, cv=5, scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
        print("Finished cross validation for LR")
        cvMSE = np.mean(abs(cvScores["test_neg_mean_squared_error"])).round(3)
        cvRMSE = np.mean(abs(cvScores["test_neg_root_mean_squared_error"])).round(3)
        cvMAE = np.mean(abs(cvScores["test_neg_mean_absolute_error"])).round(3)
        print("Average CV test scores for LR:")
        print("MSE: " + str(cvMSE))
        print("RMSE: " + str(cvRMSE))
        print("MAE: " + str(cvMAE))
        print("Finished displaying cross validation scores for LR")
        myLR.fit(xTrain, yTrain, sample_weight=sampleWeights)
        print("Finished training LR")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        print("Finished displaying training metrics for LR")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for newly-trained LR")
        pickle.dump(myLR, open("InterpModels/LRModel.pickle", "wb"))
        print("LR Model Saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

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
        myRF = pickle.load(open("InterpModels/RFModel.pickle", "rb"))
        print("RF Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for loaded RF")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        myRF = RandomForestRegressor()
        # Perform cross validation
        print("Performing cross validation for RF")
        cvScores = cross_validate(myRF, xTrain, yTrain.values.ravel(), cv=5, scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
        print("Finished cross validation for RF")
        cvMSE = np.mean(abs(cvScores["test_neg_mean_squared_error"])).round(3)
        cvRMSE = np.mean(abs(cvScores["test_neg_root_mean_squared_error"])).round(3)
        cvMAE = np.mean(abs(cvScores["test_neg_mean_absolute_error"])).round(3)
        print("Average CV test scores for RF:")
        print("MSE: " + str(cvMSE))
        print("RMSE: " + str(cvRMSE))
        print("MAE: " + str(cvMAE))
        print("Finished displaying cross validation scores for RF")
        myRF.fit(xTrain, yTrain.values.ravel(), sample_weight=sampleWeights)
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for newly-trained RF")
        pickle.dump(myRF, open("InterpModels/RFModel.pickle", "wb"))
        print("RF Model Saved")
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalARIMA(window, testWindow, loadModel=False):
    xTrain, yTrain, xTest, yTest = makeTrainTest("ARIMA", window, testWindow)
    
    # ARIMA works with the time series directly, so we'll use yTrain as the series
    # We'll try different ARIMA orders and pick the best one based on AIC
    # Common starting point is ARIMA(1,1,1) but we'll search a small grid
    
    if loadModel:
        myARIMA = pickle.load(open("InterpModels/ARIMAModel.pickle", "rb"))
        best_order = myARIMA.model.order
        print("ARIMA Model Loaded")
        print(f"ARIMA order: {best_order}")
    else:
        # Grid search for best ARIMA parameters
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # Search over a reasonable parameter space
        p_range = range(0, 4)  # AR order
        d_range = range(0, 2)  # Differencing order
        q_range = range(0, 4)  # MA order
        
        print("Searching for best ARIMA parameters...")
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(yTrain.values.flatten(), order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.3f}")
        myARIMA = best_model
        
        # Perform cross-validation using rolling window
        print("Performing cross validation for ARIMA")
        cv_mse_scores = []
        cv_rmse_scores = []
        cv_mae_scores = []
        
        # 5-fold time series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(yTrain):
            train_cv = yTrain.iloc[train_idx].values.flatten()
            val_cv = yTrain.iloc[val_idx].values.flatten()
            
            try:
                cv_model = ARIMA(train_cv, order=best_order)
                cv_fitted = cv_model.fit()
                cv_pred = cv_fitted.forecast(steps=len(val_cv))
                
                cv_mse_scores.append(mean_squared_error(val_cv, cv_pred))
                cv_rmse_scores.append(np.sqrt(mean_squared_error(val_cv, cv_pred)))
                cv_mae_scores.append(np.mean(np.abs(val_cv - cv_pred)))
            except:
                continue
        
        cvMSE = round(np.mean(cv_mse_scores), 3) if cv_mse_scores else 0
        cvRMSE = round(np.mean(cv_rmse_scores), 3) if cv_rmse_scores else 0
        cvMAE = round(np.mean(cv_mae_scores), 3) if cv_mae_scores else 0
        
        print("Average CV test scores for ARIMA:")
        print("MSE: " + str(cvMSE))
        print("RMSE: " + str(cvRMSE))
        print("MAE: " + str(cvMAE))
        print("Finished cross validation for ARIMA")
    
    # Create a wrapper class for ARIMA to have a predict method compatible with getModelMetrics
    class ARIMAWrapper:
        def __init__(self, model, train_data, order):
            self.model = model
            self.train_data = train_data
            self.order = order
        
        def predict(self, steps):
            if isinstance(steps, pd.DataFrame) or isinstance(steps, np.ndarray):
                n_steps = len(steps)
            else:
                n_steps = steps
            return self.model.forecast(steps=n_steps)
        
        def refit(self, new_data):
            """Refit the model with new data"""
            model = ARIMA(new_data.values.flatten(), order=self.order)
            self.model = model.fit()
            return self
    
    if loadModel:
        wrapper = ARIMAWrapper(myARIMA, yTrain, myARIMA.model.order)
    else:
        wrapper = ARIMAWrapper(myARIMA, yTrain, best_order)
    
    # Get training metrics
    train_pred = wrapper.predict(len(yTrain))
    trainR2 = round(r2_score(yTrain.values.flatten(), train_pred), 3)
    trainAdjR2 = 1 - (1-trainR2)*(len(yTrain)-1)/(len(yTrain)-sum(best_order if not loadModel else myARIMA.model.order)-1)
    trainAdjR2 = round(trainAdjR2, 3)
    trainMSE = round(mean_squared_error(yTrain.values.flatten(), train_pred), 3)
    trainRMSE = round(np.sqrt(trainMSE), 3)
    trainMAE = round(np.mean(np.abs(yTrain.values.flatten() - train_pred)), 3)
    trainCorr = round(np.corrcoef(train_pred, yTrain.values.flatten())[0,1], 3)
    
    print("Training Metrics for ARIMA:")
    print("R^2: " + str(trainR2))
    print("Adjusted R^2: " + str(trainAdjR2))
    print("MSE: " + str(trainMSE))
    print("RMSE: " + str(trainRMSE))
    print("MAE: " + str(trainMAE))
    print("Pearson's Correlation Coefficient: " + str(trainCorr))
    
    # Get testing metrics
    test_pred = wrapper.predict(len(yTest))
    testR2 = round(r2_score(yTest.values.flatten(), test_pred), 3)
    testAdjR2 = 1 - (1-testR2)*(len(yTest)-1)/(len(yTest)-sum(best_order if not loadModel else myARIMA.model.order)-1)
    testAdjR2 = round(testAdjR2, 3)
    testMSE = round(mean_squared_error(yTest.values.flatten(), test_pred), 3)
    testRMSE = round(np.sqrt(testMSE), 3)
    testMAE = round(np.mean(np.abs(yTest.values.flatten() - test_pred)), 3)
    testCorr = round(np.corrcoef(test_pred, yTest.values.flatten())[0,1], 3)
    
    print("Testing Metrics for ARIMA:")
    print("MSE: " + str(testMSE))
    print("RMSE: " + str(testRMSE))
    print("MAE: " + str(testMAE))
    print("Pearson's Correlation Coefficient: " + str(testCorr))
    
    if not loadModel:
        pickle.dump(myARIMA, open("InterpModels/ARIMAModel.pickle", "wb"))
        print("ARIMA Model Saved")
        return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        # For loaded models, return dummy CV metrics (0) to match the expected format
        return 0, 0, 0, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

# Reference: https://stackoverflow.com/questions/62393032/custom-loss-function-with-weights-in-keras
def myMSE(weights):
    def mseCalcs(y_true, y_pred):
        error = y_true-y_pred
        return keras.backend.mean(keras.backend.square(error) * weights)
    return mseCalcs

def trainEvalNN(window, testWindow, loadModel=False):
    if os.path.exists("nnProject"):
        shutil.rmtree("nnProject")
        print("Old NN project deleted")
    xTrain, yTrain, xTest, yTest = makeTrainTest("NN", window, testWindow, secondTime=loadModel)
    if not loadModel:
        sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    else:
        if window != 346:
            sampleWeights = tf.constant((pd.Series(getSampleWeights(testWindow, testWindow))), dtype=tf.float32)
        else:
            sampleWeights = tf.constant((pd.Series(getSampleWeights(6, 6))), dtype=tf.float32)
    if loadModel:
        # myNN = keras.models.load_model("InterpModels/NNModel.h5", custom_objects={'loss': myMSE(sampleWeights)})
        myNN = keras.models.load_model("InterpModels/NNModel.h5", compile=False)
        myNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        print("Loaded NN")
        history = myNN.fit(
        xTrain, yTrain,
        validation_data=(xTest, yTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        sample_weight=sampleWeights
        )
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myNN, "NN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myNN, "NN", training=False)
        print(myNN.summary())
        print("Finished displaying testing metrics for loaded NN")
        myNN.save("InterpModels/NNModel.h5")
        print("NN Model Saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        tuner = RandomSearch(
        lambda hp: buildNN(sampleWeights, hp),
        objective = 'val_loss',
        max_trials = 300, # 300
        executions_per_trial = 3, #3
        directory = "nnProject",
        project_name = "NN"
        )
        # print(tuner.search_space_summary())
        tuner.search(xTrain, yTrain, epochs=100, validation_data=(xTest, yTest), callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)], sample_weight = sampleWeights)
        print(tuner.results_summary())
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        myNN = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        history = myNN.fit(
        xTrain, yTrain,
        validation_data=(xTest, yTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        sample_weight = sampleWeights
        )
        print("Finished training NN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myNN, "NN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myNN, "NN", training=False)
        print(myNN.summary())
        print("Finished displaying testing metrics for newly-trained NN")
        myNN.save("InterpModels/NNModel.h5")
        print("NN Model Saved")
    # if trainMAE is a Pandas series, convert it to a float
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildNN(sampleWeights, hp):
    myNN = keras.Sequential()
    myNN.add(layers.Dense(units = 24, activation='relu', input_shape=[23]))
    for i in range(hp.Int('layers', 1, 5)):
        myNN.add(layers.Dense(units=hp.Int('units_' + str(i), 2, 60, step=2),
                                        activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid']),
                                        kernel_regularizer=keras.regularizers.l2(hp.Choice('l2_' + str(i), [0.01, 0.001, 0.0001]))))
        myNN.add(layers.Dropout(hp.Float('dropout_' + str(i), 0.2, 0.7, step=0.05)))
    myNN.add(layers.Dense(1))
    myNN.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myNN

def trainEvalRNN(window, testWindow, loadModel=False):
    if os.path.exists("rnnProject"):
        shutil.rmtree("rnnProject")
        print("Old RNN project deleted")
    xTrain, yTrain, xTest, yTest = makeTrainTest("RNN", window, testWindow, secondTime=loadModel)
    sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
    timestep = 12 # number of timesteps to look back
    # Split the xTrain and xTeset into rolling windows of size timestep, and have yTrain and yTest be the next value
    rnnXTrain = np.array([xTrain[i:i+timestep] for i in range(len(xTrain)-timestep)])
    rnnXTest = np.array([xTest[i:i+timestep] for i in range(testWindow)])
    rnnYTrain = np.array([yTrain.values[i+timestep] for i in range(len(yTrain)-timestep)])
    rnnYTest = np.array([yTest.values[i] for i in range(len(yTest))])
    rnnXTrain.reshape(len(rnnXTrain), timestep, 23)
    rnnXTest.reshape(len(rnnXTest), timestep, 23)
    if not loadModel:
        sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    else:
        if window != 346:
            sampleWeights = tf.constant((pd.Series(getSampleWeights(testWindow, testWindow))), dtype=tf.float32)
        else:
            sampleWeights = tf.constant((pd.Series(getSampleWeights(6, 6))), dtype=tf.float32)
    if len(sampleWeights) > len(rnnXTrain):
        sampleWeights = sampleWeights[:len(rnnXTrain)]
    if loadModel:
        # myRNN = keras.models.load_model("InterpModels/RNNModel.h5", custom_objects={'loss': myMSE(sampleWeights)})
        myRNN = keras.models.load_model("InterpModels/RNNModel.h5", compile=False)
        myRNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        print("Loaded RNN")
        history = myRNN.fit(
        rnnXTrain, rnnYTrain,
        validation_data=(rnnXTest, rnnYTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        sample_weight=sampleWeights
        )
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for loaded RNN")
        myRNN.save("InterpModels/RNNModel.h5")
        print("RNN saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        tuner = RandomSearch(
        lambda hp: buildRNN(sampleWeights, hp),
        objective = 'val_loss',
        max_trials = 300, #  300
        executions_per_trial = 3,
        directory = "rnnProject",
        project_name = "RNN"
        )
        # print(tuner.search_space_summary())
        tuner.search(rnnXTrain, rnnYTrain, epochs=100, validation_data=(rnnXTest, rnnYTest), callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)], sample_weight = sampleWeights)
        print(tuner.results_summary())
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        myRNN = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        history = myRNN.fit(
        rnnXTrain, rnnYTrain,
        validation_data=(rnnXTest, rnnYTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        sample_weight = sampleWeights
        )
        print("Finished training RNN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for newly-trained RNN")
        myRNN.save("InterpModels/RNNModel.h5")
        print("RNN saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildRNN(sampleWeights, hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.SimpleRNN(units = hp.Int('units', 1, 60), activation='relu', input_shape=(12,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def makeModel(myNN):
    NN = Sequential()
    for count, layer in enumerate(myNN.layers):
        if count == 0:
            NN.add(layers.Dense(layer.units, activation=layer.activation, input_shape=[23]))
        elif "dense" in layer.name:
            NN.add(layers.Dense(layer.units, activation=layer.activation))
        else:
            NN.add(layers.Dropout(layer.rate))
    NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return NN

def trainEvalEnsemble(window, testWindow, loadModel=False):
    xTrain, yTrain, xTest, yTest = makeTrainTest("Ensemble", window, testWindow)
    sampleWeights = getSampleWeights(window, testWindow)
    # Load in the individual models
    if loadModel:
        myLR = pickle.load(open("InterpModels/EnsembleModel.pickle", "rb"))
        print("Ensemble Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "Ensemble", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "Ensemble", training=False)
        print("Finished displaying testing metrics for loaded Ensemble")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        myLR = pickle.load(open("InterpModels/LRModel.pickle", "rb"))
        myRF = pickle.load(open("InterpModels/RFModel.pickle", "rb"))
        myNN = keras.models.load_model("InterpModels/NNModel.h5", compile=False)
        myNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        sciNN = KerasRegressor(model = makeModel(myNN), epochs=100, batch_size=32, verbose=0, optimizer="adam")
        myEnsemble = VotingRegressor(estimators=[('LR', myLR), ('RF', myRF), ('NN', sciNN)])
        myEnsemble.fit(xTrain, yTrain, sample_weight=sampleWeights)
        print("Finished training ensemble")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myEnsemble, "Ensemble", training=True)
        print("Displayed training metrics for ensemble")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myEnsemble, "Ensemble", training=False)
        print("Displayed testing metrics for ensemble")
        pickle.dump(myEnsemble, open("InterpModels/EnsembleModel.pickle", "wb"))
        print("Saved ensemble model")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate inflation forecasting models (INTERPOLATED DATA)')
    parser.add_argument('--models', nargs='+', choices=['LR', 'RF', 'ARIMA', 'NN', 'RNN', 'Ensemble', 'all'],
                        default=['all'], help='Specify which models to train (default: all)')
    parser.add_argument('--load-model', action='store_true', help='Load existing models instead of training from scratch')
    args = parser.parse_args()
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = {'LR', 'RF', 'ARIMA', 'NN', 'RNN', 'Ensemble'}
    else:
        models_to_train = set(args.models)
    
    loadModel = args.load_model
    print(f"Training models: {', '.join(sorted(models_to_train))}")
    print(f"Load existing models: {loadModel}")
    
    pre2020trainWindowSize, pre2020testWindowSize = [490 + 20*i for i in range(49)], 20
    pre2020trainMetrics, pre2020testMetrics = [], []
    transTrainWindowSize, transTestWindowSize = [1470], 16
    transTrainMetrics, transTestMetrics = [], []
    post2020trainWindowSize, post2020testWindowSize = [1486 + 16*i for i in range(9)], 16
    # Note: window 1630 didn't work so cutting it back to just to get things working
    # last window so with the yTest it is off by 1 value (yTest has 15 values instead of 16)
    # because there isn't a 16th point for it to get?
    post2020trainMetrics, post2020testMetrics = [], []
    for count, window in enumerate(pre2020trainWindowSize):
        if count == 0:
            metricsDict = {}
            if 'LR' in models_to_train:
                metricsDict["LR"+str(window)] = trainEvalLR(window, pre2020testWindowSize, loadModel)
            if 'RF' in models_to_train:
                metricsDict["RF"+str(window)] = trainEvalRF(window, pre2020testWindowSize, loadModel)
            if 'ARIMA' in models_to_train:
                metricsDict["ARIMA"+str(window)] = trainEvalARIMA(window, pre2020testWindowSize, loadModel)
            if 'NN' in models_to_train:
                metricsDict["NN"+str(window)] = trainEvalNN(window, pre2020testWindowSize, loadModel)
            if 'RNN' in models_to_train:
                metricsDict["RNN"] = trainEvalRNN(window, pre2020testWindowSize, loadModel)
            if 'Ensemble' in models_to_train:
                metricsDict["Ensemble"] = trainEvalEnsemble(window, pre2020testWindowSize, loadModel)
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
        else:
            metricsDict = {}
            if 'LR' in models_to_train:
                metricsDict["LR"+str(window)] = trainEvalLR(window, pre2020testWindowSize, loadModel)
            if 'RF' in models_to_train:
                metricsDict["RF"+str(window)] = trainEvalRF(window, pre2020testWindowSize, loadModel)
            if 'ARIMA' in models_to_train:
                metricsDict["ARIMA"+str(window)] = trainEvalARIMA(window, pre2020testWindowSize, loadModel=True)
            if 'NN' in models_to_train:
                metricsDict["NN"+str(window)] = trainEvalNN(window, pre2020testWindowSize, loadModel=True)
            if 'RNN' in models_to_train:
                metricsDict["RNN"] = trainEvalRNN(window, pre2020testWindowSize, loadModel=True)
            if 'Ensemble' in models_to_train:
                metricsDict["Ensemble"] = trainEvalEnsemble(window, pre2020testWindowSize, loadModel)
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
    for window in transTrainWindowSize:
        metricsDict = {}
        if 'LR' in models_to_train:
            metricsDict["LR"+str(window)] = trainEvalLR(window, transTestWindowSize, loadModel)
        if 'RF' in models_to_train:
            metricsDict["RF"+str(window)] = trainEvalRF(window, transTestWindowSize, loadModel)
        if 'ARIMA' in models_to_train:
            metricsDict["ARIMA"+str(window)] = trainEvalARIMA(window, transTestWindowSize, loadModel=True)
        if 'NN' in models_to_train:
            metricsDict["NN"+str(window)] = trainEvalNN(window, transTestWindowSize, loadModel=True)
        if 'RNN' in models_to_train:
            metricsDict["RNN"] = trainEvalRNN(window, transTestWindowSize, loadModel=True)
        if 'Ensemble' in models_to_train:
            metricsDict["Ensemble"] = trainEvalEnsemble(window, transTestWindowSize, loadModel)
        train, test = compileMetrics(metricsDict)
        transTrainMetrics.append(train)
        transTestMetrics.append(test)
    for count, window in enumerate(post2020trainWindowSize):
        metricsDict = {}
        if 'LR' in models_to_train:
            metricsDict["LR"+str(window)] = trainEvalLR(window, post2020testWindowSize, loadModel)
        if 'RF' in models_to_train:
            metricsDict["RF"+str(window)] = trainEvalRF(window, post2020testWindowSize, loadModel)
        if 'ARIMA' in models_to_train:
            metricsDict["ARIMA"+str(window)] = trainEvalARIMA(window, post2020testWindowSize, loadModel=True)
        if 'NN' in models_to_train:
            metricsDict["NN"+str(window)] = trainEvalNN(window, post2020testWindowSize, loadModel=True)
        if 'RNN' in models_to_train:
            metricsDict["RNN"] = trainEvalRNN(window, post2020testWindowSize, loadModel=True)
        if 'Ensemble' in models_to_train:
            metricsDict["Ensemble"] = trainEvalEnsemble(window, post2020testWindowSize, loadModel)
        train, test = compileMetrics(metricsDict)
        post2020trainMetrics.append(train)
        post2020testMetrics.append(test)
    
    pre2020train, pre2020test = pd.concat(pre2020trainMetrics), pd.concat(pre2020testMetrics)
    transtrain, transtest = pd.concat(transTrainMetrics), pd.concat(transTestMetrics)
    post2020train, post2020test = pd.concat(post2020trainMetrics), pd.concat(post2020testMetrics)
    
    # Add average rows for each model type that was trained
    if 'LR' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "LR"
        pre2020train.loc['Pre2020LRavg'] = pre2020train.loc[pre2020train.index.str.contains("LR")].mean()
        pre2020test.loc['Pre2020LRavg'] = pre2020test.loc[pre2020test.index.str.contains("LR")].mean()
        transtrain.loc['TransLRavg'] = transtrain.loc[transtrain.index.str.contains("LR")].mean()
        transtest.loc['TransLRavg'] = transtest.loc[transtest.index.str.contains("LR")].mean()
        post2020train.loc['Post2020LRavg'] = post2020train.loc[post2020train.index.str.contains("LR")].mean()
        post2020test.loc['Post2020LRavg'] = post2020test.loc[post2020test.index.str.contains("LR")].mean()
    
    if 'RF' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "RF"
        pre2020train.loc['Pre2020RFavg'] = pre2020train.loc[pre2020train.index.str.contains("RF")].mean()
        pre2020test.loc['Pre2020RFavg'] = pre2020test.loc[pre2020test.index.str.contains("RF")].mean()
        transtrain.loc['TransRFavg'] = transtrain.loc[transtrain.index.str.contains("RF")].mean()
        transtest.loc['TransRFavg'] = transtest.loc[transtest.index.str.contains("RF")].mean()
        post2020train.loc['Post2020RFavg'] = post2020train.loc[post2020train.index.str.contains("RF")].mean()
        post2020test.loc['Post2020RFavg'] = post2020test.loc[post2020test.index.str.contains("RF")].mean()
    
    if 'ARIMA' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "ARIMA"
        pre2020train.loc['Pre2020ARIMAavg'] = pre2020train.loc[pre2020train.index.str.contains("ARIMA")].mean()
        pre2020test.loc['Pre2020ARIMAavg'] = pre2020test.loc[pre2020test.index.str.contains("ARIMA")].mean()
        transtrain.loc['TransARIMAavg'] = transtrain.loc[transtrain.index.str.contains("ARIMA")].mean()
        transtest.loc['TransARIMAavg'] = transtest.loc[transtest.index.str.contains("ARIMA")].mean()
        post2020train.loc['Post2020ARIMAavg'] = post2020train.loc[post2020train.index.str.contains("ARIMA")].mean()
        post2020test.loc['Post2020ARIMAavg'] = post2020test.loc[post2020test.index.str.contains("ARIMA")].mean()
    
    if 'NN' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "NN"
        pre2020train.loc['Pre2020NNavg'] = pre2020train.loc[pre2020train.index.str.contains("NN")].mean()
        pre2020test.loc['Pre2020NNavg'] = pre2020test.loc[pre2020test.index.str.contains("NN")].mean()
        transtrain.loc['TransNNavg'] = transtrain.loc[transtrain.index.str.contains("NN")].mean()
        transtest.loc['TransNNavg'] = transtest.loc[transtest.index.str.contains("NN")].mean()
        post2020train.loc['Post2020NNavg'] = post2020train.loc[post2020train.index.str.contains("NN")].mean()
        post2020test.loc['Post2020NNavg'] = post2020test.loc[post2020test.index.str.contains("NN")].mean()
    
    if 'RNN' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "RNN"
        pre2020train.loc['Pre2020RNNavg'] = pre2020train.loc[pre2020train.index.str.contains("RNN")].mean()
        pre2020test.loc['Pre2020RNNavg'] = pre2020test.loc[pre2020test.index.str.contains("RNN")].mean()
        transtrain.loc['TransRNNavg'] = transtrain.loc[transtrain.index.str.contains("RNN")].mean()
        transtest.loc['TransRNNavg'] = transtest.loc[transtest.index.str.contains("RNN")].mean()
        post2020train.loc['Post2020RNNavg'] = post2020train.loc[post2020train.index.str.contains("RNN")].mean()
        post2020test.loc['Post2020RNNavg'] = post2020test.loc[post2020test.index.str.contains("RNN")].mean()
    
    if 'Ensemble' in models_to_train:
        # add a row at the end which is the average of each column for rows containing "Ensemble"
        pre2020train.loc['Pre2020Ensembleavg'] = pre2020train.loc[pre2020train.index.str.contains("Ensemble")].mean()
        pre2020test.loc['Pre2020Ensembleavg'] = pre2020test.loc[pre2020test.index.str.contains("Ensemble")].mean()
        transtrain.loc['TransEnsembleavg'] = transtrain.loc[transtrain.index.str.contains("Ensemble")].mean()
        transtest.loc['TransEnsembleavg'] = transtest.loc[transtest.index.str.contains("Ensemble")].mean()
        post2020train.loc['Post2020Ensembleavg'] = post2020train.loc[post2020train.index.str.contains("Ensemble")].mean()
        post2020test.loc['Post2020Ensembleavg'] = post2020test.loc[post2020test.index.str.contains("Ensemble")].mean()
    
    # combine the train and test dataframes into one
    train = pd.concat([pre2020train,transtrain, post2020train])
    test = pd.concat([pre2020test, transtest, post2020test])
    # move rows that contain "avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("avg")],train[train.index.str.contains("avg")]])
    test = pd.concat([test[~test.index.str.contains("avg")],test[test.index.str.contains("avg")]])
    
    # START OF TOTAL AVERAGE CODE - Calculate weighted averages across all time periods
    if 'LR' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020LRavg")]*20 
        temp2 = train.loc[train.index.str.contains('TransLRavg')]*16
        temp3 = train.loc[train.index.str.contains("Post2020LRavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020LRavg")]*20
        temp2 = test.loc[test.index.str.contains("TransLRavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020LRavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    if 'RF' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020RFavg")]*20
        temp2 = train.loc[train.index.str.contains("TransRFavg")]*16
        temp3 = train.loc[train.index.str.contains("Post2020RFavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020RFavg")]*20
        temp2 = test.loc[test.index.str.contains("TransRFavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020RFavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    if 'ARIMA' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020ARIMAavg")]*20
        temp2 = train.loc[train.index.str.contains("TransARIMAavg")]*16
        temp3 = train.loc[train.index.str.contains("Post2020ARIMAavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalARIMAAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020ARIMAavg")]*20
        temp2 = test.loc[test.index.str.contains("TransARIMAavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020ARIMAavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalARIMAAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    if 'NN' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020NNavg")]*20
        temp2 = train.loc[train.index.str.contains("TransNNavg")]*16
        temp3 = train.loc[train.index.str.contains("Post2020NNavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020NNavg")]*20
        temp2 = test.loc[test.index.str.contains("TransNNavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020NNavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    if 'RNN' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020RNNavg")]*20
        temp2 = train.loc[train.index.str.contains("TransRNNavg")]*16
        temp3 = train.loc[train.index.str.contains("Post2020RNNavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020RNNavg")]*20
        temp2 = test.loc[test.index.str.contains("TransRNNavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020RNNavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    if 'Ensemble' in models_to_train:
        temp = train.loc[train.index.str.contains("Pre2020Ensembleavg")]*20
        temp2 = train.loc[train.index.str.contains("TransEnsembleavg")]*16
        temp3 = train.loc[train.index.str.contains("Post2020Ensembleavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
        train = pd.concat([train, temp])
        temp = test.loc[test.index.str.contains("Pre2020Ensembleavg")]*20
        temp2 = test.loc[test.index.str.contains("TransEnsembleavg")]*16
        temp3 = test.loc[test.index.str.contains("Post2020Ensembleavg")]*16
        for i in range(len(temp2.columns)):
            temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
        temp = temp/52
        temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
        test = pd.concat([test, temp])
    
    # move the rows that contain "Avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("Avg")],train[train.index.str.contains("Avg")]])
    test = pd.concat([test[~test.index.str.contains("Avg")],test[test.index.str.contains("Avg")]])
    
    # Generate filename based on models trained
    if len(models_to_train) == 6:  # All models trained
        filename_suffix = "AllModels"
    else:
        # Sort models alphabetically for consistent naming
        model_names = sorted(list(models_to_train))
        filename_suffix = "_".join(model_names)
    
    # save the dataframes to excel files
    train_filename = f"Metrics/InterpTrainMetricsEVERY4_{filename_suffix}.xlsx"
    test_filename = f"Metrics/InterpTestMetricsEVERY4_{filename_suffix}.xlsx"
    train.to_excel(train_filename)
    test.to_excel(test_filename)
    print(f"Results saved to:")
    print(f"  - {train_filename}")
    print(f"  - {test_filename}")
    print("Program Done")
    # mainPlotting()

if __name__ == "__main__":
    main()

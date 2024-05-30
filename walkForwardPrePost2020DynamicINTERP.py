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
# import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# from keras.api.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from keras import layers, metrics, Sequential, models, regularizers
from keras_tuner.tuners import RandomSearch
from sklearn.linear_model import Lasso
import shutil, os
import pickle
from functools import partial
import argparse
from sklearn.model_selection import cross_validate, TimeSeriesSplit
# from yellowbrick.regressor.residuals import residuals_plot
# from scikeras.wrappers import KerasRegressor
# from sklearn.ensemble import VotingRegressor
from generateMetricPlots import mainPlotting

from customLR import myLR

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, window, testWindow, secondTime=False): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    econData = readEconData("Data/ConstructedDataframes/INTERPAllEcon1990AndCOVIDWithLagsDynamic.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # Drop NaN columns
    # econData.dropna(axis=1, inplace=True)
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
    xTrain, yTrain, xTest, yTest = xTrain.dropna(axis=1, inplace=False), yTrain.dropna(axis=1, inplace=False), xTest.dropna(axis=1, inplace=False), yTest.dropna(axis=1, inplace=False)
    # Drop any columns in xTest and yTest that are not in xTrain and yTrain
    xTest = xTest[xTrain.columns]
    yTest = yTest[yTrain.columns]
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
    predictions = model.predict(x)
    r2 = round(r2_score(y, predictions),3)
    try:
        adjR2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
    except ZeroDivisionError:
        adjR2 = 0
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
    trainingMetrics = pd.DataFrame(columns = ["cvMSE", "cvRMSE", "cvMAE","Train R^2", "Train Adjusted R^2", "Train MSE", "Train RMSE", "Train MAE", "Train Pearson's Correlation Coefficient"])
    testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
    for key in metricsDict.keys():
        if "LR" in key or "RF" in key:
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
        myLRModel = myLR.load("Models/dynamicLRModel.pickle")#pickle.load(open("Models/dynamicLRModel.pickle", "rb"))
        print("LR Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLRModel, "LR", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLRModel, "LR", training=False)
        print("Finished displaying testing metrics for loaded LR")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
    else:
        myLRModel = myLR() #LinearRegression()
        # Perform cross validation
        # print("Performing cross validation for LR")
        # cvScores = cross_validate(myLRModel, xTrain, yTrain, cv=5, scoring=["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
        # print("Finished cross validation for LR")
        # cvMSE = np.mean(abs(cvScores["test_neg_mean_squared_error"])).round(3)
        # cvRMSE = np.mean(abs(cvScores["test_neg_root_mean_squared_error"])).round(3)
        # cvMAE = np.mean(abs(cvScores["test_neg_mean_absolute_error"])).round(3)
        # print("Average CV test scores for LR:")
        # print("MSE: " + str(cvMSE))
        # print("RMSE: " + str(cvRMSE))
        # print("MAE: " + str(cvMAE))
        # print("Finished displaying cross validation scores for LR")
        myLRModel.fit(xTrain, yTrain)#, sample_weight=sampleWeights)
        print("Finished training LR")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLRModel, "LR", training=True)
        print("Finished displaying training metrics for LR")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLRModel, "LR", training=False)
        print("Finished displaying testing metrics for newly-trained LR")
        myLRModel.save("Models/dynamicLRModel.pickle")#pickle.dump(myLRModel, open("Models/dynamicLRModel.pickle", "wb"))
        print("LR Model Saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
        # return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
        return -999, -999, -999, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr
def trainEvalRF(window, testWindow, loadModel=False):
    xTrain, yTrain, xTest, yTest = makeTrainTest("RF", window, testWindow)
    sampleWeights = getSampleWeights(window, testWindow)
    if loadModel:
        myRF = pickle.load(open("Models/RFModel.pickle", "rb"))
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
        pickle.dump(myRF, open("Models/RFModel.pickle", "wb"))
        print("RF Model Saved")
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

# Reference: https://stackoverflow.com/questions/62393032/custom-loss-function-with-weights-in-keras
def myMSE(weights):
    def mseCalcs(y_true, y_pred):
        error = y_true-y_pred
        return keras.backend.mean(keras.backend.square(error) * weights)
    return mseCalcs


def extendNN(nn, numNewFeatures):
    newNN = keras.Sequential()
    oldNumFeatures = nn.layers[0].units
    newNN.add(layers.Dense(units = oldNumFeatures+numNewFeatures, activation='relu', input_shape=[oldNumFeatures+numNewFeatures]))
    
    # Create new layers that are copies of the old ones
    for layer in nn.layers[1:]:
        # If the layer is a dense layer, add a new dense layer to the new network
        if isinstance(layer, layers.Dense):
            newLayer = layers.Dense(units=layer.units, activation=layer.activation)
        # If the layer is a dropout layer, add a new dropout layer to the new network
        elif isinstance(layer, layers.Dropout):
            newLayer = layers.Dropout(rate=layer.rate)
        newNN.add(newLayer)
    
    # Copy the weights from the old network to the new one, only creating new weights for the input layer
    for i in range(len(nn.layers)):
        if i == 0:
            weights = nn.layers[i].get_weights()
            newWeights = np.random.rand(oldNumFeatures + numNewFeatures, oldNumFeatures + numNewFeatures)
            newWeights[:oldNumFeatures, :weights[0].shape[1]] = weights[0]
            newBiases = np.random.rand(oldNumFeatures + numNewFeatures)
            newBiases[:weights[1].shape[0]] = weights[1]
            newNN.layers[i].set_weights([newWeights, newBiases])
        elif i == 1:
            weights = nn.layers[i].get_weights()
            newWeights = np.random.rand(oldNumFeatures + numNewFeatures, nn.layers[i].units)
            newWeights[:oldNumFeatures, :] = weights[0]
            newNN.layers[i].set_weights([newWeights, weights[1]])    
        else:
            newNN.layers[i].set_weights(nn.layers[i].get_weights())
    return newNN


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
        # myNN = keras.models.load_model("Models/NNModel.h5", custom_objects={'loss': myMSE(sampleWeights)})
        myNN = keras.models.load_model("Models/NNModel.h5", compile=False)
        print("Loaded NN")
        # If the new train data has more features than the old model (check iunput shape), extend the model to have the new features
        print('NN input shape', myNN.input_shape[1])
        print('xTrain shape', xTrain.shape[1])
        if xTrain.shape[1] > myNN.input_shape[1]:
            myNN = extendNN(myNN, xTrain.shape[1] - myNN.input_shape[1])
            print("Extended NN")
        myNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        print("Compiled NN")
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
        myNN.save("Models/NNModel.h5")
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
        myNN.save("Models/NNModel.h5")
        print("NN Model Saved")
    # if trainMAE is a Pandas series, convert it to a float
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildNN(sampleWeights, hp):
    myNN = keras.Sequential()
    myNN.add(layers.Dense(units = 19, activation='relu', input_shape=[19]))
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
        # myRNN = keras.models.load_model("Models/RNNModel.h5", custom_objects={'loss': myMSE(sampleWeights)})
        myRNN = keras.models.load_model("Models/RNNModel.h5", compile=False)
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
        myRNN.save("Models/RNNModel.h5")
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
        myRNN.save("Models/RNNModel.h5")
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
        myLR = pickle.load(open("Models/EnsembleModel.pickle", "rb"))
        print("Ensemble Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "Ensemble", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "Ensemble", training=False)
        print("Finished displaying testing metrics for loaded Ensemble")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        myLR = pickle.load(open("Models/LRModel.pickle", "rb"))
        myRF = pickle.load(open("Models/RFModel.pickle", "rb"))
        myNN = keras.models.load_model("Models/NNModel.h5", compile=False)
        myNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        sciNN = KerasRegressor(model = makeModel(myNN), epochs=100, batch_size=32, verbose=0, optimizer="adam")
        myEnsemble = VotingRegressor(estimators=[('LR', myLR), ('RF', myRF), ('NN', sciNN)])
        myEnsemble.fit(xTrain, yTrain, sample_weight=sampleWeights)
        print("Finished training ensemble")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myEnsemble, "Ensemble", training=True)
        print("Displayed training metrics for ensemble")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myEnsemble, "Ensemble", training=False)
        print("Displayed testing metrics for ensemble")
        pickle.dump(myEnsemble, open("Models/EnsembleModel.pickle", "wb"))
        print("Saved ensemble model")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    loadModel = False
    pre2020trainWindowSize, pre2020testWindowSize = [490 + 20*i for i in range(49)], 20
    pre2020trainMetrics, pre2020testMetrics = [], []
    transTrainWindowSize, transTestWindowSize = [1470], 16
    transTrainMetrics, transTestMetrics = [], []
    post2020trainWindowSize, post2020testWindowSize = [1486 + 16*i for i in range(9)], 16
    post2020trainMetrics, post2020testMetrics = [], []
    for count, window in enumerate(pre2020trainWindowSize):
        if count == 0:
            metricsDict = {"LR"+str(window): trainEvalLR(window,pre2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window, pre2020testWindowSize, loadModel), "NN"+str(window): trainEvalNN(window, pre2020testWindowSize, loadModel)}#, "RNN": trainEvalRNN(window, pre2020testWindowSize, loadModel), "Ensemble": trainEvalEnsemble(window, pre2020testWindowSize, loadModel) }
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
        else:
            metricsDict = {"LR"+str(window): trainEvalLR(window,pre2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window, pre2020testWindowSize, loadModel), "NN"+str(window): trainEvalNN(window, pre2020testWindowSize, loadModel=True)}#, "RNN": trainEvalRNN(window, pre2020testWindowSize, loadModel=True), "Ensemble": trainEvalEnsemble(window, pre2020testWindowSize, loadModel) }
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
    for window in transTrainWindowSize:
        metricsDict = {"LR"+str(window): trainEvalLR(window,transTestWindowSize, loadModel), "RF"+str(window): trainEvalRF(window, transTestWindowSize, loadModel), "NN"+str(window): trainEvalNN(window, transTestWindowSize, loadModel=True)}#, "RNN": trainEvalRNN(window, transTestWindowSize, loadModel=True), "Ensemble": trainEvalEnsemble(window, transTestWindowSize, loadModel) }
        train, test = compileMetrics(metricsDict)
        transTrainMetrics.append(train)
        transTestMetrics.append(test)
    for count, window in enumerate(post2020trainWindowSize):
        metricsDict = {"LR"+str(window): trainEvalLR(window,post2020testWindowSize, loadModel), "RF"+str(window): trainEvalRF(window, post2020testWindowSize, loadModel), "NN"+str(window): trainEvalNN(window, post2020testWindowSize, loadModel=True)}#, "RNN": trainEvalRNN(window, post2020testWindowSize, loadModel=True), "Ensemble": trainEvalEnsemble(window, post2020testWindowSize, loadModel) }
        train, test = compileMetrics(metricsDict)
        post2020trainMetrics.append(train)
        post2020testMetrics.append(test)
    
    pre2020train, pre2020test = pd.concat(pre2020trainMetrics), pd.concat(pre2020testMetrics)
    transtrain, transtest = pd.concat(transTrainMetrics), pd.concat(transTestMetrics)
    post2020train, post2020test = pd.concat(post2020trainMetrics), pd.concat(post2020testMetrics)
    # add a row at the end which is the average of each column for rows containing "LR"
    pre2020train.loc['Pre2020LRavg'] = pre2020train.loc[pre2020train.index.str.contains("LR")].mean()
    pre2020test.loc['Pre2020LRavg'] = pre2020test.loc[pre2020test.index.str.contains("LR")].mean()
    transtrain.loc['TransLRavg'] = transtrain.loc[transtrain.index.str.contains("LR")].mean()
    transtest.loc['TransLRavg'] = transtest.loc[transtest.index.str.contains("LR")].mean()
    post2020train.loc['Post2020LRavg'] = post2020train.loc[post2020train.index.str.contains("LR")].mean()
    post2020test.loc['Post2020LRavg'] = post2020test.loc[post2020test.index.str.contains("LR")].mean()
    # add a row at the end which is the average of each column for rows containing "RF"
    pre2020train.loc['Pre2020RFavg'] = pre2020train.loc[pre2020train.index.str.contains("RF")].mean()
    pre2020test.loc['Pre2020RFavg'] = pre2020test.loc[pre2020test.index.str.contains("RF")].mean()
    transtrain.loc['TransRFavg'] = transtrain.loc[transtrain.index.str.contains("RF")].mean()
    transtest.loc['TransRFavg'] = transtest.loc[transtest.index.str.contains("RF")].mean()
    post2020train.loc['Post2020RFavg'] = post2020train.loc[post2020train.index.str.contains("RF")].mean()
    post2020test.loc['Post2020RFavg'] = post2020test.loc[post2020test.index.str.contains("RF")].mean()
    # add a row at the end which is the average of each column for rows containing "NN"
    pre2020train.loc['Pre2020NNavg'] = pre2020train.loc[pre2020train.index.str.contains("NN")].mean()
    pre2020test.loc['Pre2020NNavg'] = pre2020test.loc[pre2020test.index.str.contains("NN")].mean()
    transtrain.loc['TransNNavg'] = transtrain.loc[transtrain.index.str.contains("NN")].mean()
    transtest.loc['TransNNavg'] = transtest.loc[transtest.index.str.contains("NN")].mean()
    post2020train.loc['Post2020NNavg'] = post2020train.loc[post2020train.index.str.contains("NN")].mean()
    post2020test.loc['Post2020NNavg'] = post2020test.loc[post2020test.index.str.contains("NN")].mean()
    # add a row at the end which is the average of each column for rows containing "RNN"
    pre2020train.loc['Pre2020RNNavg'] = pre2020train.loc[pre2020train.index.str.contains("RNN")].mean()
    pre2020test.loc['Pre2020RNNavg'] = pre2020test.loc[pre2020test.index.str.contains("RNN")].mean()
    transtrain.loc['TransRNNavg'] = transtrain.loc[transtrain.index.str.contains("RNN")].mean()
    transtest.loc['TransRNNavg'] = transtest.loc[transtest.index.str.contains("RNN")].mean()
    post2020train.loc['Post2020RNNavg'] = post2020train.loc[post2020train.index.str.contains("RNN")].mean()
    post2020test.loc['Post2020RNNavg'] = post2020test.loc[post2020test.index.str.contains("RNN")].mean()
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
    train = pd.concat([pre2020train,transtrain, post2020train])
    test = pd.concat([pre2020test, transtest, post2020test])# START OF TOTAL AVERAGE CODE
    temp = train.loc[train.index.str.contains("Pre2020LRavg")]*10 
    temp2 = train.loc[train.index.str.contains('TransLRavg')]*5
    temp3 = train.loc[train.index.str.contains("Post2020LRavg")]*2
    # Add each value of dataframe temp2 to temp's values, leading to temp's values array being of shape (1, 9)
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    # Divide temp's values by 12
    temp = temp/17
    # Rename the index of temp to TotalLRAvg
    temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
    # Add temp as a row to the end of the train dataframe
    train = pd.concat([train, temp])
    # Rename the last row in train to TotalLRAvg
    # train.rename(index={train.index[-1]:'TotalLRAvg'}, inplace=True)
    # Repeat the above steps for the test dataframe
    temp = test.loc[test.index.str.contains("Pre2020LRavg")]*10
    temp2 = test.loc[test.index.str.contains("TransLRavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020LRavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalLRAvg
    temp.rename(index={temp.index[0]:'TotalLRAvg'}, inplace=True)
    test = pd.concat([test, temp])
    test.rename(index={test.index[-1]:'TotalLRAvg'}, inplace=True)
    # Repeat the above steps for the RF model
    temp = train.loc[train.index.str.contains("Pre2020RFavg")]*10
    temp2 = train.loc[train.index.str.contains("TransRFavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020RFavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRFAvg
    temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRFAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020RFavg")]*10
    temp2 = test.loc[test.index.str.contains("TransRFavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020RFavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRFAvg
    temp.rename(index={temp.index[0]:'TotalRFAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalRFAvg'}, inplace=True)
    # Repeat the above steps for the NN model
    temp = train.loc[train.index.str.contains("Pre2020NNavg")]*10
    temp2 = train.loc[train.index.str.contains("TransNNavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020NNavg")]*10
    temp2 = test.loc[test.index.str.contains("TransNNavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalNNAvg'}, inplace=True)
    # Repeat the above steps for the RNN model
    temp = train.loc[train.index.str.contains("Pre2020RNNavg")]*10
    temp2 = train.loc[train.index.str.contains("TransRNNavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020RNNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020RNNavg")]*10
    temp2 = test.loc[test.index.str.contains("TransRNNavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020RNNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalRNNAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalRNNAvg'}, inplace=True)
    # Repeat the above steps for the Ensemble model
    temp = train.loc[train.index.str.contains("Pre2020Ensembleavg")]*10
    temp2 = train.loc[train.index.str.contains("TransEnsembleavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020Ensembleavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalEnsembleAvg
    temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalEnsembleAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020Ensembleavg")]*10
    temp2 = test.loc[test.index.str.contains("TransEnsembleavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020Ensembleavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalEnsembleAvg
    temp.rename(index={temp.index[0]:'TotalEnsembleAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalEnsembleAvg'}, inplace=True)
    # move the rows that contain "Avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("avg")],train[train.index.str.contains("avg")]])
    test = pd.concat([test[~test.index.str.contains("avg")],test[test.index.str.contains("avg")]])
    # save the dataframes to excel files
    train.to_excel("Metrics/InterprollingTrainMetricsDynamic.xlsx")
    test.to_excel("Metrics/InterprollingTestMetricsDynamic.xlsx")
    print("Program Done")
    # mainPlotting()

if __name__ == "__main__":
    main()

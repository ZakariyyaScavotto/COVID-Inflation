import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
# import to hide the annoying warnings about numba.jit when importing shap library
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras import layers, metrics, Sequential
from keras_tuner.tuners import RandomSearch
import shutil, os
from generateMetricPlots import mainPlotting
"""RUN IN ANACONDA gpuTime ENVIRONMENT TO USE GPU!!!"""

timestep = 6 # number of timesteps to look back

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
            xTest = econData.iloc[window-timestep:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
            yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        else:
            if window == 346: #First post2020
                trainDf = econData.iloc[window-6-timestep:window]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest = econData.iloc[window-timestep:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
                yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
            else:
                trainDf = econData.iloc[window-testWindow-timestep:window]
                xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                xTest = econData.iloc[window-timestep:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
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

# Reference: https://stackoverflow.com/questions/62393032/custom-loss-function-with-weights-in-keras
def myMSE(weights):
    def mseCalcs(y_true, y_pred):
        error = y_true-y_pred
        return keras.backend.mean(keras.backend.square(error) * weights)
    return mseCalcs

def trainEvalRNN(window, testWindow, loadModel=False):
    if os.path.exists("rnnProject"):
        shutil.rmtree("rnnProject")
        print("Old RNN project deleted")
    xTrain, yTrain, xTest, yTest = makeTrainTest("RNN", window, testWindow, secondTime=loadModel)
    sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
    
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

def trainEvalLSTM(window, testWindow, loadModel=False):
    if os.path.exists("lstmProject"):
        shutil.rmtree("lstmProject")
        print("Old LSTM project deleted")
    xTrain, yTrain, xTest, yTest = makeTrainTest("RNN", window, testWindow, secondTime=loadModel)
    sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
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
        myRNN = keras.models.load_model("Models/LSTMModel.h5", compile=False)
        myRNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        print("Loaded LSTM")
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
        myRNN.save("Models/LSTMModel.h5")
        print("LSTM saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        tuner = RandomSearch(
        lambda hp: buildLSTM(sampleWeights, hp),
        objective = 'val_loss',
        max_trials = 300, #  300
        executions_per_trial = 3,
        directory = "lstmProject",
        project_name = "LSTM"
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
        print("Finished training LSTM")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for newly-trained LSTM")
        myRNN.save("Models/LSTMModel.h5")
        print("LSTM saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalGRU(window, testWindow, loadModel=False):
    if os.path.exists("gruProject"):
        shutil.rmtree("gruProject")
        print("Old GRU project deleted")
    xTrain, yTrain, xTest, yTest = makeTrainTest("RNN", window, testWindow, secondTime=loadModel)
    sampleWeights = tf.constant((pd.Series(getSampleWeights(window, testWindow))), dtype=tf.float32)
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
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
        myRNN = keras.models.load_model("Models/GRUModel.h5", compile=False)
        myRNN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        print("Loaded GRU")
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
        print("Finished displaying testing metrics for loaded GRU")
        myRNN.save("Models/GRUModel.h5")
        print("GRU saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    else:
        tuner = RandomSearch(
        lambda hp: buildGRU(sampleWeights, hp),
        objective = 'val_loss',
        max_trials = 300, #  300
        executions_per_trial = 3,
        directory = "gruProject",
        project_name = "GRU"
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
        print("Finished training GRU")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for newly-trained GRU")
        myRNN.save("Models/GRUModel.h5")
        print("GRU saved")
        if isinstance(trainMAE, pd.Series):
            trainMAE = trainMAE[0]
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildRNN(sampleWeights, hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.SimpleRNN(units = hp.Int('units', 1, 60), activation='relu', input_shape=(timestep,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def buildLSTM(sampleWeights, hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.LSTM(units = hp.Int('units', 1, 60), activation='relu', input_shape=(timestep,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def buildGRU(sampleWeights, hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.GRU(units = hp.Int('units', 1, 60), activation='relu', input_shape=(timestep,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def main():
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available")
    else:
        print("GPU not available")
    loadModel = False
    pre2020trainWindowSize, pre2020testWindowSize = [490 + 20*i for i in range(49)], 20
    pre2020trainMetrics, pre2020testMetrics = [], []
    transTrainWindowSize, transTestWindowSize = [1470], 16
    transTrainMetrics, transTestMetrics = [], []
    post2020trainWindowSize, post2020testWindowSize = [1486 + 16*i for i in range(9)], 16
    post2020trainMetrics, post2020testMetrics = [], []
    for count, window in enumerate(pre2020trainWindowSize):
        if count == 0:
            metricsDict = {"LSTM": trainEvalLSTM(window, pre2020testWindowSize, loadModel), "GRU": trainEvalGRU(window, pre2020testWindowSize, loadModel), "RNN": trainEvalRNN(window, pre2020testWindowSize, loadModel)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
        else:
            metricsDict = {"LSTM": trainEvalLSTM(window, pre2020testWindowSize, loadModel=True), "GRU": trainEvalGRU(window, pre2020testWindowSize, loadModel=True), "RNN": trainEvalRNN(window, pre2020testWindowSize, loadModel=True)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
    for window in transTrainWindowSize:
        metricsDict = {"LSTM": trainEvalLSTM(window, transTestWindowSize, loadModel=True), "GRU": trainEvalGRU(window, transTestWindowSize, loadModel=True), "RNN": trainEvalRNN(window, transTestWindowSize, loadModel=True)}
        train, test = compileMetrics(metricsDict)
        transTrainMetrics.append(train)
        transTestMetrics.append(test)
    for count, window in enumerate(post2020trainWindowSize):
        metricsDict = {"LSTM": trainEvalLSTM(window, post2020testWindowSize, loadModel=True), "GRU": trainEvalGRU(window, post2020testWindowSize, loadModel=True), "RNN": trainEvalRNN(window, post2020testWindowSize, loadModel=True)}
        train, test = compileMetrics(metricsDict)
        post2020trainMetrics.append(train)
        post2020testMetrics.append(test)
    
    pre2020train, pre2020test = pd.concat(pre2020trainMetrics), pd.concat(pre2020testMetrics)
    transtrain, transtest = pd.concat(transTrainMetrics), pd.concat(transTestMetrics)
    post2020train, post2020test = pd.concat(post2020trainMetrics), pd.concat(post2020testMetrics)
    
    # add a row at the end which is the average of each column for rows containing "LSTM"
    pre2020train.loc['Pre2020LSTMavg'] = pre2020train.loc[pre2020train.index.str.contains("LSTM")].mean()
    pre2020test.loc['Pre2020LSTMavg'] = pre2020test.loc[pre2020test.index.str.contains("LSTM")].mean()
    transtrain.loc['TransLSTMavg'] = transtrain.loc[transtrain.index.str.contains("LSTM")].mean()
    transtest.loc['TransLSTMavg'] = transtest.loc[transtest.index.str.contains("LSTM")].mean()
    post2020train.loc['Post2020LSTMavg'] = post2020train.loc[post2020train.index.str.contains("LSTM")].mean()
    post2020test.loc['Post2020LSTMavg'] = post2020test.loc[post2020test.index.str.contains("LSTM")].mean()
    # add a row at the end which is the average of each column for rows containing "GRU"
    pre2020train.loc['Pre2020GRUavg'] = pre2020train.loc[pre2020train.index.str.contains("GRU")].mean()
    pre2020test.loc['Pre2020GRUavg'] = pre2020test.loc[pre2020test.index.str.contains("GRU")].mean()
    transtrain.loc['TransGRUavg'] = transtrain.loc[transtrain.index.str.contains("GRU")].mean()
    transtest.loc['TransGRUavg'] = transtest.loc[transtest.index.str.contains("GRU")].mean()
    post2020train.loc['Post2020GRUavg'] = post2020train.loc[post2020train.index.str.contains("GRU")].mean()
    post2020test.loc['Post2020GRUavg'] = post2020test.loc[post2020test.index.str.contains("GRU")].mean()
    # add a row at the end which is the average of each column for rows containing "RNN"
    pre2020train.loc['Pre2020RNNavg'] = pre2020train.loc[pre2020train.index.str.contains("RNN")].mean()
    pre2020test.loc['Pre2020RNNavg'] = pre2020test.loc[pre2020test.index.str.contains("RNN")].mean()
    transtrain.loc['TransRNNavg'] = transtrain.loc[transtrain.index.str.contains("RNN")].mean()
    transtest.loc['TransRNNavg'] = transtest.loc[transtest.index.str.contains("RNN")].mean()
    post2020train.loc['Post2020RNNavg'] = post2020train.loc[post2020train.index.str.contains("RNN")].mean()
    post2020test.loc['Post2020RNNavg'] = post2020test.loc[post2020test.index.str.contains("RNN")].mean()
    # combine the train and test dataframes into one
    train = pd.concat([pre2020train,transtrain, post2020train])
    test = pd.concat([pre2020test, transtest, post2020test])
    # move rows that contain "avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("avg")], train[train.index.str.contains("avg")]])
    test = pd.concat([test[~test.index.str.contains("avg")], test[test.index.str.contains("avg")]])
    # START OF TOTAL AVERAGE CODE
    
    # Repeat the above steps for the LSTM model
    temp = train.loc[train.index.str.contains("Pre2020LSTMavg")]*10
    temp2 = train.loc[train.index.str.contains("TransLSTMavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020LSTMavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalLSTMAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020LSTMavg")]*10
    temp2 = test.loc[test.index.str.contains("TransLSTMavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020LSTMavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalLSTMAvg'}, inplace=True)
    test = pd.concat([test, temp])

    # Repeat the above steps for the GRU model
    temp = train.loc[train.index.str.contains("Pre2020GRUavg")]*10
    temp2 = train.loc[train.index.str.contains("TransGRUavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020GRUavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalGRUAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalRNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020GRUavg")]*10
    temp2 = test.loc[test.index.str.contains("TransGRUavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020GRUavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalRNNAvg
    temp.rename(index={temp.index[0]:'TotalGRUAvg'}, inplace=True)
    test = pd.concat([test, temp])

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
    # move the rows that contain "Avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("Avg")], train[train.index.str.contains("Avg")]])
    test = pd.concat([test[~test.index.str.contains("Avg")], test[test.index.str.contains("Avg")]])
    # save the dataframes to excel files
    train.to_excel("Metrics/sixMonthTimestepTrainINTERP.xlsx")
    test.to_excel("Metrics/sixMonthTimestepTestINTERP.xlsx")
    print("Program Done")
    mainPlotting()

if __name__ == "__main__":
    main()

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
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, metrics
from keras_tuner.tuners import RandomSearch
from sklearn.linear_model import Lasso
from regressors import stats, plots
import shutil, os
import pickle
from functools import partial

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    econData = readEconData("Data\ConstructedDataframes\AllEcon1990AndCOVIDWithLags.xlsx")
    # econData = readEconData("Data\ConstructedDataframes\ALLECONDATAwithLagsAndCOVIDDataANDInflationLag.xlsx")
    # econData = readEconData("Data\ConstructedDataframes\AutoregressiveSigLags.xlsx")
    if modelName in ["LR", "NN", "RF"]:
        # drop the date column
        econData.drop("Date", axis=1, inplace=True)
        # scale the data using StandardScaler
        scaler = StandardScaler()
        econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        # split into train/test using sklearn train_test_split
        trainDf, testDf = train_test_split(econData, test_size=0.2, random_state=42)
        # split into x and y
        xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
        xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
        return xTrain, yTrain, xTest, yTest
    else:
        # split into pre-2020 and 2020->beyond
        ind2020 = econData.query("Date == '2020-01-01'").index.values[0]
        pre2020EconData, post2020EconData = econData.iloc[:ind2020].copy(), econData.iloc[ind2020:].copy()
        # now, can remove the date column
        pre2020EconData.drop("Date", axis=1, inplace=True)
        post2020EconData.drop("Date", axis=1, inplace=True)
        # now scale the data
        scaler, scaler2 = StandardScaler(), StandardScaler()
        pre2020EconData = pd.DataFrame(scaler.fit_transform(pre2020EconData), columns=pre2020EconData.columns)
        post2020EconData = pd.DataFrame(scaler2.fit_transform(post2020EconData), columns=post2020EconData.columns)
        # now need to split each into train/test
        # split pre-2020
        trainCutoff = int(pre2020EconData.shape[0] * 0.8) 
        pre2020TrainDf, pre2020TestDf = pre2020EconData.iloc[:trainCutoff], pre2020EconData.iloc[trainCutoff:]
        pre2020xTrain, pre2020yTrain = pre2020TrainDf.loc[:, pre2020TrainDf.columns != "AnnualizedMoM-CPI-Inflation"], pre2020TrainDf.loc[:, pre2020TrainDf.columns == "AnnualizedMoM-CPI-Inflation"]
        pre2020xTest, pre2020yTest = pre2020TestDf.loc[:, pre2020TestDf.columns != "AnnualizedMoM-CPI-Inflation"], pre2020TestDf.loc[:, pre2020TestDf.columns == "AnnualizedMoM-CPI-Inflation"]
        # Now do the same for post2020EconData
        trainCutoff = int(post2020EconData.shape[0] * 0.8)
        post2020TrainDf, post2020TestDf = post2020EconData.iloc[:trainCutoff], post2020EconData.iloc[trainCutoff:]
        post2020xTrain, post2020yTrain = post2020TrainDf.loc[:, post2020TrainDf.columns != "AnnualizedMoM-CPI-Inflation"], post2020TrainDf.loc[:, post2020TrainDf.columns == "AnnualizedMoM-CPI-Inflation"]
        post2020xTest, post2020yTest = post2020TestDf.loc[:, post2020TestDf.columns != "AnnualizedMoM-CPI-Inflation"], post2020TestDf.loc[:, post2020TestDf.columns == "AnnualizedMoM-CPI-Inflation"]
        return pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest

def plotPredictions(xTest, yTest, model, modelName):
    # Plot the predictions of the model on the xTest data
    predictions = model.predict(xTest)
    plt.plot(predictions, label="Predictions")
    plt.plot(yTest, label="Actual")
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
    if training and modelName != "RNN":
        mae = np.mean(np.abs(predictions - y)).values[0].round(3)
    else:
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
    if modelName ==  "LR" and training:
        printLRCoeffSig(x, y.values.tolist(), model, x.columns)
        plotLRResiduals(x, y.values.tolist(), model)
        print("Displayed LR residuals plot for training data")
        getShapPlot(x, model, modelName)
        print("Displayed LR SHAP plot for training data")
    elif modelName == "LR" and not training:
        plotLRResiduals(x, y.tolist(), model)
        print("Displayed LR residuals plot for testing data")
    elif modelName == "RF" and training:
        getShapPlot(x, model, modelName)
        print("Displayed RF SHAP plot for training data")
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
    # for each model, the metrics are stored in a list in the order of R^2, Adjusted R^2, MSE, RMSE, MAE, and Pearson's Correlation Coefficient
    # for each item in metricsDict, the key is the model name and the value is the list of metrics
    # the first 6 values are the training values, and the second 6 values are the testing values
    trainingMetrics = pd.DataFrame(columns=["Train R^2", "Train Adjusted R^2", "Train MSE", "Train RMSE", "Train MAE", "Train Pearson's Correlation Coefficient"])
    testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
    for key in metricsDict.keys():
        trainingMetrics.loc[key] = metricsDict[key][:6]
        testingMetrics.loc[key] = metricsDict[key][6:]
    trainingMetrics.to_excel("Metrics/TrainingMetrics.xlsx")
    testingMetrics.to_excel("Metrics/TestingMetrics.xlsx")
    print("Metrics saved to Excel files")
    return trainingMetrics, testingMetrics

def trainEvalLR(loadModel=False):
    '''    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # combine the xTrains into one dataframe, the xTests into one dataframe, the yTrains into one dataframe, and the yTests into one dataframe
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    '''
    xTrain, yTrain, xTest, yTest = makeTrainTest("LR")
    if loadModel:
        myLR = pickle.load(open("Models/LRModel.pickle", "rb"))
        print("LR Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for loaded LR")
    else:
        myLR = LinearRegression()
        myLR.fit(xTrain, yTrain)
        print("Finished training LR")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
        print("Finished displaying training metrics for LR")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
        print("Finished displaying testing metrics for newly-trained LR")
        pickle.dump(myLR, open("Models/LRModel.pickle", "wb"))
        print("LR Model Saved")
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def printLRCoeffSig(xTrain, yTrain, LR, xColumns):
    yTrain = [arr[0] for arr in yTrain]
    yTrain = np.array(yTrain)
    LR.coef_ = LR.coef_[0]
    LR.intercept_ = LR.intercept_[0]
    print(stats.summary(LR, xTrain, yTrain, xColumns))

def plotLRResiduals(xTrain, yTrain, LR):
    if any(isinstance(el, np.ndarray) or isinstance(el, list) for el in yTrain):
        yTrain = [arr[0] for arr in yTrain]
    yTrain = np.array(yTrain)
    plots.plot_residuals(LR, xTrain, yTrain, r_type="standardized")

def trainEvalRF(loadModel=False):
    '''    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # combine the xTrains into one dataframe, the xTests into one dataframe, the yTrains into one dataframe, and the yTests into one dataframe
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    '''
    xTrain, yTrain, xTest, yTest = makeTrainTest("RF")
    if loadModel:
        myRF = pickle.load(open("Models/RFModel.pickle", "rb"))
        print("RF Model Loaded")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for loaded RF")
    else:
        '''
        myRF = RandomForestRegressor(warm_start=True)
        myRF.fit(pre2020xTrain, pre2020yTrain.values.ravel())
        print("Finished training RF with pre2020 data")
        getModelMetrics(pre2020xTrain, pre2020yTrain, myRF, "RF", training=True)
        getModelMetrics(pre2020xTest, pre2020yTest, myRF, "RF", training=False)
        print("Now training RF with 2020->beyond data")
        myRF.n_estimators += 100
        
        myRF.fit(post2020xTrain, post2020yTrain.values.ravel())
        print("Finished training RF with post2020 data")
        '''
        myRF = RandomForestRegressor()
        myRF.fit(xTrain, yTrain.values.ravel())
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
        print("Finished displaying testing metrics for newly-trained RF")
        pickle.dump(myRF, open("Models/RFModel.pickle", "wb"))
        print("RF Model Saved")
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

'''
Upon searching around for auto hyperparameter setting, came across keras-tuner
After doing some reading to understand how it works, I found this article to be helpful in learning how to implement it
https://www.analyticsvidhya.com/blog/2021/06/keras-tuner-auto-neural-network-architecture-selection/
'''

def newTrainEvalNN(loadModel=False):
    if os.path.exists("nnProject"):
        shutil.rmtree("nnProject")
        print("Old NN project deleted")
    '''    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # combine the xTrains into one dataframe, the xTests into one dataframe, the yTrains into one dataframe, and the yTests into one dataframe
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    '''
    xTrain, yTrain, xTest, yTest = makeTrainTest("NN")
    if loadModel:
        myNN = keras.models.load_model("Models/NNModel.h5")
        print("Loaded NN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myNN, "NN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myNN, "NN", training=False)
        print(myNN.summary())
        print("Finished displaying testing metrics for loaded NN")
    else:
        tuner = RandomSearch(
        buildNN,
        objective = 'val_loss',
        max_trials = 300, # 300
        executions_per_trial = 3, #3
        directory = "nnProject",
        project_name = "NN"
        )
        # print(tuner.search_space_summary())
        tuner.search(xTrain, yTrain, epochs=100, validation_data=(xTest, yTest))
        print(tuner.results_summary())
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        myNN = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        history = myNN.fit(
        xTrain, yTrain,
        validation_data=(xTest, yTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        #callbacks=[es]
        )
        print("Finished training NN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myNN, "NN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myNN, "NN", training=False)
        print(myNN.summary())
        print("Finished displaying testing metrics for newly-trained NN")
        myNN.save("Models/NNModel.h5")
        print("NN Model Saved")
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildNN(hp):
    myNN = keras.Sequential()
    myNN.add(layers.Dense(units = 24, activation='relu', input_shape=[23]))
    for i in range(hp.Int('layers', 1, 5)):
        myNN.add(layers.Dense(units=hp.Int('units_' + str(i), 2, 60, step=2),
                                        activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid']),
                                        kernel_regularizer=keras.regularizers.l2(hp.Choice('l2_' + str(i), [0.01, 0.001, 0.0001]))))
        myNN.add(layers.Dropout(hp.Float('dropout_' + str(i), 0.2, 0.7, step=0.05)))
    myNN.add(layers.Dense(1))
    myNN.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = 'mse', metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myNN

def trainEvalRNN(loadModel=False):
    if os.path.exists("rnnProject"):
        shutil.rmtree("rnnProject")
        print("Old RNN project deleted")
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest("RNN")
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
    timestep = 12 # number of timesteps to look back
    # Split the xTrain and xTeset into rolling windows of size timestep, and have yTrain and yTest be the next value
    rnnXTrain = np.array([xTrain[i:i+timestep] for i in range(len(xTrain)-timestep)])
    rnnXTest = np.array([xTest[i:i+timestep] for i in range(len(xTest)-timestep)])
    rnnYTrain = np.array([yTrain.values[i+timestep] for i in range(len(yTrain)-timestep)])
    rnnYTest = np.array([yTest.values[i+timestep] for i in range(len(yTest)-timestep)])
    rnnXTrain.reshape(len(rnnXTrain), timestep, 23)
    rnnXTest.reshape(len(rnnXTest), timestep, 23)
    if loadModel:
        myRNN = keras.models.load_model("Models/RNNModel.h5")
        print("Loaded RNN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for loaded RNN")
    else:
        tuner = RandomSearch(
        buildRNN,
        objective = 'val_loss',
        max_trials = 300, #  300
        executions_per_trial = 3,
        directory = "rnnProject",
        project_name = "RNN"
        )
        # print(tuner.search_space_summary())
        tuner.search(rnnXTrain, rnnYTrain, epochs=100, validation_data=(rnnXTest, rnnYTest))
        print(tuner.results_summary())
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        myRNN = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        history = myRNN.fit(
        rnnXTrain, rnnYTrain,
        validation_data=(rnnXTest, rnnYTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        #callbacks=[es]
        )
        print("Finished training RNN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        print(myRNN.summary())
        print("Finished displaying testing metrics for newly-trained RNN")
        myRNN.save("Models/RNNModel.h5")
        print("RNN saved")
    return trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def buildRNN(hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.SimpleRNN(units = hp.Int('units', 1, 60), activation='relu', input_shape=(12,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = 'mse', metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def buildRNNTime(time, hp):
    myRNN = keras.Sequential()
    myRNN.add(layers.SimpleRNN(units = hp.Int('units', 1, 60), activation='relu', input_shape=(time,23), dropout = (hp.Float('dropout', 0.2, 0.7, step=0.05)), recurrent_dropout = (hp.Float('recurDropout' , 0.2, 0.7, step=0.05)),return_sequences=False))
    myRNN.add(layers.Dense(1))
    myRNN.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-4])),
                    loss = 'mse', metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return myRNN

def determineRNNTimestep():
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest("RNN")
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    # xTrain, xTest = xTrain.values.reshape(-1, 1, 24), xTest.values.reshape(-1, 1, 24) # reshape train/test data for RNN to work (adding time dimension)
    bestTimestep = -1
    bestRMSE= 999999.9999999999
    for timestep in range(3, 13):
        if os.path.exists("rnnProject"):
            shutil.rmtree("rnnProject")
            print("Old RNN project deleted")
        # Split the xTrain and xTeset into rolling windows of size timestep, and have yTrain and yTest be the next value
        rnnXTrain = np.array([xTrain[i:i+timestep] for i in range(len(xTrain)-timestep)])
        rnnXTest = np.array([xTest[i:i+timestep] for i in range(len(xTest)-timestep)])
        rnnYTrain = np.array([yTrain.values[i+timestep] for i in range(len(yTrain)-timestep)])
        rnnYTest = np.array([yTest.values[i+timestep] for i in range(len(yTest)-timestep)])
        rnnXTrain.reshape(len(rnnXTrain), timestep, 23)
        rnnXTest.reshape(len(rnnXTest), timestep, 23)
        # buildRNNModel = partial(buildRNN, time=timestep)
        tuner = RandomSearch(
        lambda hp : buildRNNTime(timestep, hp),
        objective = 'val_loss',
        max_trials = 200, #  300
        executions_per_trial = 3,
        directory = "rnnProject",
        project_name = "RNN"
        )
        # print(tuner.search_space_summary())
        tuner.search(rnnXTrain, rnnYTrain, epochs=100, validation_data=(rnnXTest, rnnYTest))
        print(tuner.results_summary())
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        myRNN = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
        history = myRNN.fit(
        rnnXTrain, rnnYTrain,
        validation_data=(rnnXTest, rnnYTest),
        batch_size=32,
        epochs=100,
        verbose=0, # decided to make verbose to follow the training, feel free to set to 0
        #callbacks=[es]
        )
        print("Finished training RNN")
        trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(rnnXTrain, rnnYTrain, myRNN, "RNN", training=True)
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(rnnXTest, rnnYTest, myRNN, "RNN", training=False)
        if testRMSE < bestRMSE:
            bestRMSE = testRMSE
            bestTimestep = timestep
    print("BEST RNN TIMESTEP: ", bestTimestep)

def main():
    loadModel = False
    metricsDict = {"LR": trainEvalLR(loadModel), "RF": trainEvalRF(loadModel), "NN": newTrainEvalNN(loadModel), "RNN": trainEvalRNN(loadModel)}
    compileMetrics(metricsDict)
    # determineRNNTimestep()
    print("Program Done")

if __name__ == "__main__":
    main()

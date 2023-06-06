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
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import Lasso

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    econData = readEconData("Data\ConstructedDataframes\ALLECONDATAwithLagsAndCOVIDData.xlsx")
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

def makeTrainTestOld():
    # Split the data into training and testing (80/20) while keeping time-order
    # Read econ data
    econData = readEconData("Data\ConstructedDataframes\ALLECONDATAwithLagsAndCOVIDData.xlsx")
    scaler = StandardScaler()
    if "Date" in econData.columns:
        econData.drop("Date", axis=1, inplace=True)
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    trainCutoff = int(econData.shape[0] * 0.8) 
    trainDf, testDf = econData.iloc[:trainCutoff], econData.iloc[trainCutoff:]
    xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    return xTrain, yTrain, xTest, yTest

def trainLR():
    myLR = LinearRegression()
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # combine the xTrains into one dataframe, the xTests into one dataframe, the yTrains into one dataframe, and the yTests into one dataframe
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    myLR.fit(xTrain, yTrain)
    print("Finished training LR")
    # reference: https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    explainer = shap.explainers.Linear(myLR, xTrain)
    shap_values = explainer.shap_values(xTrain)
    shap.summary_plot(shap_values, xTrain, feature_names=xTrain.columns, plot_type="bar", show=False)
    plt.title("Feature Importance for Linear Regression")
    plt.gcf().canvas.manager.set_window_title("Feature Importance for Linear Regression")
    plt.gcf().set_size_inches(10,6)
    plt.show()
    return xTest, yTest, myLR

def evaluateLR(xTest, yTest, myLR):
    # Evaluate the LR by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myLR.predict(xTest)
    print("LR MSE: ", mean_squared_error(yTest, predictions))
    print("LR R^2: ", r2_score(yTest, predictions))
    print("LR Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))

def trainEvalRF():
    myRF = RandomForestRegressor(warm_start=True)
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    myRF.fit(pre2020xTrain, pre2020yTrain.values.ravel())
    print("Finished training RF with pre2020 data")
    evaluateRF(pre2020xTest, pre2020yTest, myRF)
    print("Now training RF with 2020->beyond data")
    myRF.n_estimators += 100
    myRF.fit(post2020xTrain, post2020yTrain.values.ravel())
    print("Finished training RF with post2020 data")
    evaluateRF(post2020xTest, post2020yTest, myRF)
    # reference: https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    # combine the xTrains into one dataframe to get shap values
    xTrain = pd.concat([pre2020xTrain, post2020xTrain])
    explainer = shap.TreeExplainer(myRF)
    shap_values = explainer.shap_values(xTrain)
    shap.summary_plot(shap_values, xTrain, feature_names=xTrain.columns, plot_type="bar", show=False)
    plt.title("Feature Importance for Random Forest")
    plt.gcf().canvas.manager.set_window_title("Feature Importance for Random Forest")
    plt.gcf().set_size_inches(10,6)
    plt.show()

def evaluateRF(xTest, yTest, myRF):
    # Evaluate the RF by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myRF.predict(xTest)
    print("RF MSE: ", mean_squared_error(yTest, predictions))
    print("RF R^2: ", r2_score(yTest, predictions))
    print("RF Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))

def trainNN():
    myNN = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(300, activation='relu', input_shape=[15]),
    layers.BatchNormalization(),
    layers.Dense(150, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(75, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
    ])
    myNN.compile(
    optimizer='adam',
    loss='mse',
    )   
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    history = myNN.fit(
    xTrain, yTrain,
    validation_data=(xTest, yTest),
    batch_size=100,
    epochs=100,
    verbose=1, # decided to make verbose to follow the training, feel free to set to 0
    callbacks=[es]
    )
    print("Finished training NN")
    '''
    explainer = shap.KernelExplainer(myNN.predict, xTrain)
    shap_values = explainer.shap_values(xTrain, nsamples = "auto")
    shap.summary_plot(shap_values, xTrain, feature_names=xTrain.columns, plot_type="bar", show=False)
    plt.title("Feature Importance for NN")
    plt.gcf().canvas.manager.set_window_title("Feature Importance for NN")
    plt.gcf().set_size_inches(10,6)
    plt.show()
    '''
    return xTest, yTest, myNN

def evaluateNN(xTest, yTest, myNN):
    test_loss = myNN.evaluate(xTest, yTest, verbose=0)
    # print(myNN.metrics_names)
    print("NN Loss: ", test_loss)
    predictions = myNN.predict(xTest)
    print("NN MSE: ", mean_squared_error(yTest, predictions))
    plt.plot(range(len(yTest)), yTest, color='blue')
    plt.plot(range(len(predictions)), predictions, color='red')
    plt.legend(['Actual', 'Predicted'], loc='upper left')
    plt.title("NN Predictions vs Actual")
    plt.show()

def trainEvalLasso():
    myLasso = Lasso(alpha = 0.5, warm_start=True)
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # First, fit on pre2020 and evaluate
    myLasso.fit(pre2020xTrain, pre2020yTrain)
    print("Lasso finished first fit")
    evaluateLasso(pre2020xTest, pre2020yTest, myLasso, True)
    # now, retrain with the "new" post2020 data
    myLasso.fit(post2020xTrain, post2020yTrain)
    print("Lasso finished second fit")
    # now, evaluate in total (full test set)
    xTest = pd.concat([pre2020xTest, post2020xTest])
    yTest = pd.concat([pre2020yTest, post2020yTest])
    evaluateLasso(xTest, yTest, myLasso, False)

def evaluateLasso(xTest, yTest, myLasso, pre):
    predictions = myLasso.predict(xTest)
    if pre:
        print("Pre2020 Lasso MSE: ", mean_squared_error(yTest, predictions))
        print("Pre2020 Lasso R^2: ", r2_score(yTest, predictions))
        print("Pre2020 Lasso Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))
    else:
        print("Total Lasso MSE: ", mean_squared_error(yTest, predictions))
        print("Total Lasso R^2: ", r2_score(yTest, predictions))
        print("Total Lasso Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))

def main():
    # Try basic Lasso on the econ data
    trainEvalLasso()
    # Try basic LR on the econ data
    xTest, yTest, firstLR = trainLR()
    evaluateLR(xTest, yTest, firstLR)
    # Try basic RF on the econ data
    trainEvalRF()
    # Try basic NN on the econ data
    xTest, yTest, firstNN = trainNN()
    evaluateNN(xTest, yTest, firstNN)
    print("Program Done")

if __name__ == "__main__":
    main()

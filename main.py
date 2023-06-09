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
from regressors import stats, plots

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
    '''temp fix?'''
    if training==False:
        y = np.array([value[0] for value in y.values.tolist()])
    predictions = model.predict(x)
    r2 = r2_score(y, predictions)
    adjR2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    predictions = predictions.reshape(predictions.size, 1)
    mae = np.mean(np.abs(predictions - y))
    corr = np.corrcoef(predictions.T, y.T)[0,1]
    if training:
        print("Training Metrics for "+modelName+":")
        print("R^2: "+str(r2))
        print("Adjusted R^2: "+str(adjR2))
    else:
        plotPredictions(x, y, model, modelName)
        print("Testing Metrics for "+modelName+":")
    print("MSE: "+str(mse))
    print("RMSE: "+str(rmse))
    print("MAE: "+str(mae))
    print("Pearson's Correlation Coefficent: "+str(corr))
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

def trainEvalLR():
    myLR = LinearRegression()
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    # combine the xTrains into one dataframe, the xTests into one dataframe, the yTrains into one dataframe, and the yTests into one dataframe
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    myLR.fit(xTrain, yTrain)
    print("Finished training LR")
    getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
    print("Finished displaying training metrics for LR")
    getModelMetrics(xTest, yTest, myLR, "LR", training=False)
    print("Finished displaying testing metrics for LR")
    
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

def trainEvalRF():
    myRF = RandomForestRegressor(warm_start=True)
    pre2020xTrain, pre2020yTrain, pre2020xTest, pre2020yTest, post2020xTrain, post2020yTrain, post2020xTest, post2020yTest = makeTrainTest()
    myRF.fit(pre2020xTrain, pre2020yTrain.values.ravel())
    print("Finished training RF with pre2020 data")
    getModelMetrics(pre2020xTrain, pre2020yTrain, myRF, "RF", training=True)
    getModelMetrics(pre2020xTest, pre2020yTest, myRF, "RF", training=False)
    print("Now training RF with 2020->beyond data")
    myRF.n_estimators += 100
    myRF.fit(post2020xTrain, post2020yTrain.values.ravel())
    print("Finished training RF with post2020 data")
    xTrain, xTest, yTrain, yTest = pd.concat([pre2020xTrain, post2020xTrain]), pd.concat([pre2020xTest, post2020xTest]), pd.concat([pre2020yTrain, post2020yTrain]), pd.concat([pre2020yTest, post2020yTest])
    getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
    getModelMetrics(xTest, yTest, myRF, "RF", training=False)

def trainEvalNN():
    myNN = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(300, activation='relu', input_shape=[22]),
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
    getModelMetrics(xTrain, yTrain, myNN, "NN", training=True)
    getModelMetrics(xTest, yTest, myNN, "NN", training=False)

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
        plotPredictions(xTest, yTest, myLasso, "Lasso Pre 2020")
    else:
        print("Total Lasso MSE: ", mean_squared_error(yTest, predictions))
        print("Total Lasso R^2: ", r2_score(yTest, predictions))
        print("Total Lasso Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))
        plotPredictions(xTest, yTest, myLasso, "Lasso Full Test Set")

def main():
    # Try basic LR on the econ data
    trainEvalLR()
    # Try basic RF on the econ data
    trainEvalRF()
    # Try basic NN on the econ data
    trainEvalNN()
    # Try basic Lasso on the econ data
    trainEvalLasso()
    print("Program Done")

if __name__ == "__main__":
    main()

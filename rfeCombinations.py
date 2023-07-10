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
from sklearn.model_selection import cross_validate

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(combo): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    # econData = readEconData("Data\ConstructedDataframes\AllEcon1990AndCOVIDWithLags.xlsx")
    econData = readEconData("Data\ConstructedDataframes\AutoregressiveAllLags.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    # if the combo value for a column is 0 drop that column from econData, otherwise keep it
    droppedColumns = [column for column in econData.columns if column!= 'AnnualizedMoM-CPI-Inflation' and combo[econData.columns.get_loc(column)] == 0]
    econData.drop(droppedColumns, axis=1, inplace=True)
    # split into train/test using sklearn train_test_split
    trainDf, testDf = train_test_split(econData, test_size=0.2, random_state=42)
    # split into x and y
    xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
    return xTrain, yTrain, xTest, yTest

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
        trainingMetrics.loc[key] = metricsDict[key][:9]
        testingMetrics.loc[key] = metricsDict[key][9:]
    return trainingMetrics, testingMetrics

def trainEvalLR(combo):
    xTrain, yTrain, xTest, yTest = makeTrainTest(combo)
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
    myLR.fit(xTrain, yTrain)
    print("Finished training LR")
    trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myLR, "LR", training=True)
    print("Finished displaying training metrics for LR")
    testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myLR, "LR", training=False)
    print("Finished displaying testing metrics for newly-trained LR")
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def trainEvalRF(combo):
    xTrain, yTrain, xTest, yTest = makeTrainTest(combo)
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
    myRF.fit(xTrain, yTrain.values.ravel())
    trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr = getModelMetrics(xTrain, yTrain, myRF, "RF", training=True)
    testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(xTest, yTest, myRF, "RF", training=False)
    print("Finished displaying testing metrics for newly-trained RF")
    if isinstance(trainMAE, pd.Series):
        trainMAE = trainMAE[0]
    return cvMSE, cvRMSE, cvMAE, trainR2, trainAdjR2, trainMSE, trainRMSE, trainMAE, trainCorr, testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    LRcombos = pd.read_excel("RFE/AutoRegressiveLR.xlsx")
    RFcombos = pd.read_excel("RFE/AutoRegressiveRF.xlsx")
    # drop the first column from each dataframe
    LRcombos.drop(LRcombos.columns[0], axis=1, inplace=True)
    RFcombos.drop(RFcombos.columns[0], axis=1, inplace=True)
    trainMetrics, testMetrics = [], []
    for i in range(LRcombos.shape[0]):
        LRcombo, RFcombo = LRcombos.iloc[i], RFcombos.iloc[i]
        metricsDict = {"LR Combo"+str(i): trainEvalLR(LRcombo), "RF Combo"+str(i): trainEvalRF(RFcombo)}
        print("Finished training and evaluating LR and RF for combo " + str(i))
        train, test = compileMetrics(metricsDict)
        trainMetrics.append(train)
        testMetrics.append(test)
    trainMetrics = pd.concat(trainMetrics, axis=0)
    testMetrics = pd.concat(testMetrics, axis=0)
    # sort the train and test metrics so the LR metrics rows are next to each other and all the RF metrics rows are next to each other
    trainMetrics = trainMetrics.sort_index()
    testMetrics = testMetrics.sort_index()
    trainMetrics.to_excel("RFEMetrics/ALLAutoRegressiveTrainingMetrics.xlsx")
    testMetrics.to_excel("RFEMetrics/ALLAutoRegressiveTestingMetrics.xlsx")
    print("Program done")

if __name__ == "__main__":
    main()
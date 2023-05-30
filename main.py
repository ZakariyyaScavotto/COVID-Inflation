import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def readEconData(filename):
    df = pd.read_excel(filename)
    # Last number in HousingPriceInd missing so dropping that row
    df.dropna(subset=["HousingPriceInd"], inplace=True)
    df.drop("Date", axis=1, inplace=True)
    # print(df.isna().sum())
    return df

def makeTrainTest(econData):
    # Split the data into training and testing (80/20) while keeping time-order
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    # print(econData.head())
    trainCutoff = int(econData.shape[0] * 0.8)
    trainDf, testDf = econData.iloc[:trainCutoff], econData.iloc[trainCutoff:]
    xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    return xTrain, yTrain, xTest, yTest

def trainLR(econData):
    myLR = LinearRegression()
    xTrain, yTrain, xTest, yTest = makeTrainTest(econData)
    myLR.fit(xTrain, yTrain)
    print("Finished training LR")
    return xTest, yTest, myLR

def evaluateLR(xTest, yTest, myLR):
    # Evaluate the LR by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myLR.predict(xTest)
    print("LR MSE: ", mean_squared_error(yTest, predictions))
    print("LR R^2: ", r2_score(yTest, predictions))

def trainRF(econData):
    myRF = RandomForestRegressor()
    xTrain, yTrain, xTest, yTest = makeTrainTest(econData)
    myRF.fit(xTrain, yTrain.values.ravel())
    print("Finished training RF")
    return xTest, yTest, myRF

def evaluateRF(xTest, yTest, myRF):
    # Evaluate the RF by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myRF.predict(xTest)
    print("RF MSE: ", mean_squared_error(yTest, predictions))
    print("RF R^2: ", r2_score(yTest, predictions))


def main():
    # Read in the full econ data file
    econData = readEconData("Data\EconomicData\ALLECONDATA.xlsx")
    # Try basic LR on the econ data
    xTest, yTest, firstLR = trainLR(econData)
    evaluateLR(xTest, yTest, firstLR)
    # Try basic RF on the econ data
    xTest, yTest, firstRF = trainRF(econData)
    evaluateRF(xTest, yTest, firstRF)
    print("Program Done")


if __name__ == "__main__":
    main()

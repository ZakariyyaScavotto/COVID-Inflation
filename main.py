import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def readEconData(filename):
    df = pd.read_excel(filename)
    # Last number in HousingPriceInd missing so dropping that row
    df.dropna(subset=["HousingPriceInd"], inplace=True)
    df.drop("Date", axis=1, inplace=True)
    # print(df.isna().sum())
    return df

def makeTrainLR(econData):
    myLR = LinearRegression()
    # Split the data into training and testing (80/20) while keeping time-order
    trainCutoff = int(econData.shape[0] * 0.8)
    trainDf, testDf = econData.iloc[:trainCutoff], econData.iloc[trainCutoff:]
    xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
    myLR.fit(xTrain, yTrain)
    print("Finished training")
    return xTest, yTest, myLR

def evaluateLR(xTest, yTest, myLR):
    # Evaluate the LR by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myLR.predict(xTest)
    print("MSE: ", mean_squared_error(yTest, predictions))
    print("R^2: ", r2_score(yTest, predictions))

def main():
    # Read in the full econ data file
    econData = readEconData("Data\EconomicData\ALLECONDATA.xlsx")
    # Try basic LR on the econ data
    xTest, yTest, firstLR = makeTrainLR(econData)
    evaluateLR(xTest, yTest, firstLR)
    print("Program Done")


if __name__ == "__main__":
    main()

''' 
Create dataset with all the lags for the econ data
This file makes working in main.py easier
This file is not meant to do the train/test split or scaling, 
that is still to be done in main.py
'''
import pandas as pd

def readEconData(filename):
    df = pd.read_excel(filename)
    # Last number in HousingPriceInd missing so dropping that row
    df.dropna(subset=["HPI%Change"], inplace=True)
    return df

def readCovidData(filename):
    df = pd.read_excel(filename)
    return df

def makeLags(df, columnName, lags):
    for lag in lags:
        df[columnName + "Lag" + str(lag)] = df[columnName].shift(lag)
    df.drop(columnName, axis=1, inplace=True)
    return df

def mainDatasetMaking():
    # Read in econ data
    df = readEconData("Data\EconomicData\ALLECONDATA.xlsx")
    # Drop the BananaPrice%Change, BreadPrice%Change, EggPrice%Change, and GroundBeefPrice%Change columns from the df
    # For full reasoning see the partialAutoCorrelationPlots folder, but in sumnmary doing partial autocorrelations on
    # these columns showed that these columns are likely white noise and do not add much value to the models
    df.drop(["BananaPrice%Change", "BreadPrice%Change", "EggPrice%Change", "GroundBeefPrice%Change", "2008-9RecessionDummyVar"], axis=1, inplace=True)
    lagsDict = {"ChickenPrice%Change": [1], "ElectricityPrice%Change": [1,4], "GasolinePrice%Change": [1,2], "HouseStart%Change": [1],
                "HPI%Change": [1,2,3], "IndPro%Change": [1,2], "MichInflationExpectation": [1], "MilkPrice%Change":[1], "RentalPriceAvg%Change":[1,2,7],
                "UtilityPrice%Change": [1]}
    for colName, lags in lagsDict.items():
        df = makeLags(df, colName, lags)
    # Drop all rows with nan values from the df
    df.dropna(inplace=True)
    # Read in Covid data
    covidDf = readCovidData("Data\CovidData\ALLMONTHLYCOVIDDATA.xlsx")
    # Merge the two dataframes on the Date column
    df = pd.merge(df, covidDf, how="outer")
    # Move the AnnualizedMoM-CPI-Inflation column to the end of the df for formatting purposes
    df = df[[c for c in df if c not in ["AnnualizedMoM-CPI-Inflation"]] + ["AnnualizedMoM-CPI-Inflation"]]
    # Fill the nan values in the df with 0
    df.fillna(0, inplace=True)
    # Save the df to a csv file
    df.to_excel("Data\ConstructedDataframes\ALLECONDATAwithLagsAndCOVIDData.xlsx", index=False)
    # Save correlation matrix to excel file
    corrMatrix = df.corr()
    corrMatrix.to_excel("Data\ConstructedDataframes\CorrMatrix.xlsx")
    print("Done making and saving dataset")

if __name__ == "__main__":
    mainDatasetMaking()
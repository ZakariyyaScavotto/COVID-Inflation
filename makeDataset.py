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
    # For full reasoning see the partialAutoCorrelationPlots folder and the 6_7_23 update
    originalRemove = ["BananaPrice%Change", "BreadPrice%Change", "EggPrice%Change", "GroundBeefPrice%Change", "2008-9RecessionDummyVar"]
    nonSigRemove = ["BananaPrice%Change", "BreadPrice%Change", "EggPrice%Change", "GroundBeefPrice%Change", "2008-9RecessionDummyVar", 
                    "UnemploymentRate%Change", "ChickenPrice%Change", "HouseStart%Change", "HPI%Change", "MichInflationExpectation", 
                    "MilkPrice%Change", "UtilityPrice%Change", "2008-9RecessionDummyVar"]
    nonSigMinusMichRemove = ["BananaPrice%Change", "BreadPrice%Change", "EggPrice%Change", "GroundBeefPrice%Change", "2008-9RecessionDummyVar", 
                    "UnemploymentRate%Change", "ChickenPrice%Change", "HouseStart%Change", "HPI%Change",  
                    "MilkPrice%Change", "UtilityPrice%Change", "2008-9RecessionDummyVar"]
    # Make list of all features but AnnualizedMoM-CPI-Inflation and remove them from the df
    allRemove = ["BananaPrice%Change", "BreadPrice%Change", "EggPrice%Change", "GroundBeefPrice%Change", "2008-9RecessionDummyVar", 
                    "UnemploymentRate%Change", "ChickenPrice%Change", "HouseStart%Change", "HPI%Change", "MichInflationExpectation", 
                    "MilkPrice%Change", "UtilityPrice%Change", "2008-9RecessionDummyVar", "ElectricityPrice%Change", "GasolinePrice%Change", "IndPro%Change", "RentalPriceAvg%Change"]
    df.drop(allRemove, axis=1, inplace=True)
    # Duplicate Inflation variable to create feature for lagged inflation
    df["AnnualizedMoM-CPI-InflationFeat"] = df.loc[:,"AnnualizedMoM-CPI-Inflation"]
    originalLagsDict = {"ChickenPrice%Change": [1], "ElectricityPrice%Change": [1,4], "GasolinePrice%Change": [1,2], "HouseStart%Change": [1],
                "HPI%Change": [1,2,3], "IndPro%Change": [1,2], "MichInflationExpectation": [1, 4], "MilkPrice%Change":[1], "RentalPriceAvg%Change":[1,2,7],
                "UtilityPrice%Change": [1], "AnnualizedMoM-CPI-InflationFeat": [1]}
    nonSigRemoveSigLags = {"ElectricityPrice%Change": [4], "GasolinePrice%Change": [1, 2], "IndPro%Change": [2], "RentalPriceAvg%Change": [7]}
    nonSigMinusMichLags = {"ElectricityPrice%Change": [4], "GasolinePrice%Change": [1, 2], "IndPro%Change": [2], "RentalPriceAvg%Change": [7], "MichInflationExpectation": [1, 4]}
    allRemoveLags = {"AnnualizedMoM-CPI-InflationFeat": [i for i in range(1, 13)]}
    for colName, lags in allRemoveLags.items():
        df = makeLags(df, colName, lags)
    # Drop all rows with nan values from the df
    df.dropna(inplace=True)
    '''Commented out for non sig lags run
    # Read in Covid data
    covidDf = readCovidData("Data\CovidData\ALLMONTHLYCOVIDDATA.xlsx")
    # Merge the two dataframes on the Date column
    df = pd.merge(df, covidDf, how="outer")
    '''
    # Move the AnnualizedMoM-CPI-Inflation column to the end of the df for formatting purposes
    df = df[[c for c in df if c not in ["AnnualizedMoM-CPI-Inflation"]] + ["AnnualizedMoM-CPI-Inflation"]]
    # Fill the nan values in the df with 0
    df.fillna(0, inplace=True)
    # Save the df to a csv file
    df.to_excel("Data\ConstructedDataframes\AutoregressiveAllLags.xlsx", index=False)
    # Save correlation matrix to excel file
    corrMatrix = df.corr()
    corrMatrix.to_excel("Data\ConstructedDataframes\AutoregressiveAllLagsCorrMat.xlsx")
    print("Done making and saving dataset")

if __name__ == "__main__":
    mainDatasetMaking()
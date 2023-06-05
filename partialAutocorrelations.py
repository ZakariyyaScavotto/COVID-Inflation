import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf

def readEconData(filename):
    df = pd.read_excel(filename)
    # Last number in HousingPriceInd missing so dropping that row
    df.dropna(subset=["HPI%Change"], inplace=True)
    df.drop("Date", axis=1, inplace=True) # use for makeTrainTestOld
    return df

def plotPartialAutocorrelation(econData, columnName):
    pacfFig = plot_pacf(econData[columnName], lags=np.arange(1,12), title="Partial Autocorrelation of " + columnName)
    plt.xlabel("Months lagged")
    pacfFig.savefig('Data\partialAutocorrelationPlots\PartialAutocorrelation' + columnName + '.png')

def main():
    # Read in the full econ data file
    econData = readEconData("Data\EconomicData\ALLECONDATA.xlsx")
    for column in econData.columns:
        plotPartialAutocorrelation(econData, column)

if __name__ == "__main__":
    main()
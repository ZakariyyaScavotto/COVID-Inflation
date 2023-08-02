import pandas as pd
import numpy as np

def readData(filename):
    return pd.read_excel(filename)

def convertToMonthlyAverage(df):
    # set the index to be the first columnn
    df.set_index(df.columns[0], inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').mean()
    # Add a column for the date
    df['Date'] = df.index
    # convert the date to the first of the month
    df['Date'] = df['Date'].apply(lambda x: x.replace(day=1))
    return df

def convert():
    filename = "Data/COVIDData/UKDailyCases-OurWorldInData.xlsx"
    df = readData(filename)
    df = convertToMonthlyAverage(df)
    df.to_excel(filename)
    print(f"Converted {filename} to monthly average")

convert()
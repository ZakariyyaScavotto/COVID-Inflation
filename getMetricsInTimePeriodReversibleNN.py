import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from RevIn import RevIN
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

class nnWithReversible(nn.Module):
    def __init__(self, numDenseLayers=3, dropout=0.5):
        super(nnWithReversible, self).__init__()
        self.revInLayer = RevIN(23)
        self.dense1 = nn.Linear(23, 23)
        self.denseOut = nn.Linear(23, 1)
        self.numDenseLayers = numDenseLayers
        self.dropout = dropout
        self.double()

    def forward(self, x):
        # Use RevIn to normalize x
        x = self.revInLayer(x, 'norm')
        # Add the initial dense layer
        x = F.relu(self.dense1(x))
        for i in range(self.numDenseLayers):
            x = F.relu(self.dense1(x))
            x = nn.Dropout(p=self.dropout)(x)
        # Use RevIn to denormalize x
        x = self.revInLayer(x, 'denorm')
        # Add the final dense layer
        x = self.denseOut(x)
        # Return the output
        return x

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, start, end): # Train test but with breaking up between pre-2020 and 2020->beyond
    econData = readEconData("Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    if modelName != "RNN":
        # Filter out the data to only include the time period we want
        econData = econData[(econData["Date"] >= start) & (econData["Date"] <= end)]
        # drop the date column
        econData.drop("Date", axis=1, inplace=True)
        # scale the data using StandardScaler
        scaler = StandardScaler()
        econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        # x, y = econData.loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"], econData.loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        # return x, y
        return econData
    else:
        # Filter out the data to be from start - 12 months to end
        # subtract 12 months from start
        start = start - pd.DateOffset(months=12)
        econData = econData[(econData["Date"] >= start) & (econData["Date"] <= end)]
        # drop the date column
        econData.drop("Date", axis=1, inplace=True)
        # scale the data using StandardScaler
        scaler = StandardScaler()
        econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        # split the data into x and y
        x, y = econData.loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"], econData.loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        return x, y

def plotPredictions(xTest, yTest, model, modelName,start,end):
    # Plot the predictions of the model on the xTest data
    predictions = model(xTest)
    # convert predictions to a numpy array
    predictions = predictions.detach().numpy()
    plt.plot(predictions, label="Predictions")
    plt.plot(yTest, label="Actual", color="orange")
    plt.legend()
    plt.title("Predictions vs Actual for "+modelName)
    # Set the x axis to be the dates monthly, labeling every four months starting from start and ending at end
    dates = pd.date_range(start=start, end=end, freq="MS")
    # Set dates to be in the format of Month-Year
    dates = [date.strftime("%m-%Y") for date in dates]
    plt.xticks(np.arange(0, len(dates), 4), dates[::4], rotation=45)
    # Set the y-axis label to be the change in inflation
    plt.ylabel("Change in Annualized Month-over-month Inflation Rate")
    plt.gcf().canvas.manager.set_window_title("Predictions vs Actual for "+modelName)
    plt.show()

def getModelMetrics(x, y, model, modelName, start, end, training=True):
    # Get the R^2, Adjusted R^2, MSE, RMSE, MAE, and Pearson's Correlation Coefficent for the model
    # check if y is a numpy.ndarray
    # convert y to a numpy array
    y = np.array(y)
    # if training==False and not(y.__class__ == np.ndarray):
    #     y = np.array([value[0] for value in y.values.tolist()])
    # elif training==False:
    #     y = np.array([value[0] for value in y.tolist()])
    predictions = model(x)
    # Convert predictions, which is a tensor, to a numpy array
    predictions = predictions.detach().numpy()
    r2 = round(r2_score(y, predictions),3)
    adjR2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
    adjR2 = round(adjR2,3)
    mse = round(mean_squared_error(y, predictions),3)
    rmse = round(np.sqrt(mse),3)
    predictions = predictions.reshape(predictions.size, 1)
    mae = round(np.mean(np.abs(predictions - y)),3)
    corr = round(np.corrcoef(predictions.T, y.T)[0,1], 3)
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
        plotPredictions(x, y, model, modelName, start,end) 
    return r2, adjR2, mse, rmse, mae, corr

def compileMetrics(metricsDict, start, end, loadModel=True):
    # compile the metrics into two dataframes, the first for training metrics and the second for testing metrics
    if loadModel:
        testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
        for key in metricsDict.keys():
            testingMetrics.loc[key] = metricsDict[key]
    startString = str(start).replace(":", "-").replace("00-00-00","")
    endString = str(end).replace(":", "-").replace("00-00-00","")
    testingMetrics.to_excel("Metrics/RevNNTestingMetrics"+startString+" to "+endString+".xlsx")
    print("Metrics saved to Excel files")
    return testingMetrics

class PrepareData(Dataset):
    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
                self.X = torch.from_numpy(X)
            else:
                self.X = torch.from_numpy(X.to_numpy())
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def trainEvalRevNN(start, end, loadModel=True):
    df = makeTrainTest("RevNN", start, end) 
    xTrain, yTrain = df.loc[:, df.columns != "AnnualizedMoM-CPI-Inflation"], df.loc[:, df.columns == "AnnualizedMoM-CPI-Inflation"]
    # convert yTrain and yTest to numpy arrays for use in PrepareData
    yTrain = np.array([value[0] for value in yTrain.values.tolist()])
    trainTorch= PrepareData(X=xTrain, y=yTrain, scale_X=False)
    train_loader = DataLoader(trainTorch, batch_size=32, shuffle=True)
    trainDataset = train_loader.dataset
    x, y = trainDataset.X, trainDataset.y
    # load the model
    if loadModel:
        model = nnWithReversible()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        checkpoint = torch.load("Models/NNRevModel.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("RevNN Model Loaded")
        testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr = getModelMetrics(x, y, model, "RevNN",start,end, training=False)
        print("Finished displaying testing metrics for loaded RevNN")
        if isinstance(testMAE, pd.Series):
            testMAE = testMAE[0]
        return testR2, testAdjR2, testMSE, testRMSE, testMAE, testCorr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Starting month year in format month_year, both as numbers",  required=True)
    parser.add_argument("--end", help="Ending month year in format month_year, both as numbers", required=True)
    # args = parser.parse_args()
    #startDateTime = datetime.strptime(args.start, "%m_%Y")
    #endDateTime = datetime.strptime(args.end, "%m_%Y")
    startDateTime = datetime.strptime("1_2018", "%m_%Y")
    endDateTime = datetime.strptime("3_2023", "%m_%Y")
    # set each date's time to be 12:00:00 AM
    startDateTime = startDateTime.replace(hour=0, minute=0, second=0, microsecond=0)
    endDateTime = endDateTime.replace(hour=0, minute=0, second=0, microsecond=0)
    print("Start date: "+str(startDateTime))
    print("End date: "+str(endDateTime))
    # metricsDict = {"LR "+str(startDateTime)+" to "+str(endDateTime): trainEvalLR(startDateTime, endDateTime), "RF"+str(startDateTime)+" to "+str(endDateTime): trainEvalRF(startDateTime, endDateTime), "NN"+str(startDateTime)+" to "+str(endDateTime): trainEvalNN(startDateTime, endDateTime), "RNN"+str(startDateTime)+" to "+str(endDateTime): trainEvalRNN(startDateTime, endDateTime) ,"Ensemble"+str(startDateTime)+" to "+str(endDateTime): trainEvalEnsemble(startDateTime, endDateTime)}
    metricsDict = {"Rev NN"+str(startDateTime)+" to "+str(endDateTime): trainEvalRevNN(startDateTime, endDateTime)}
    compileMetrics(metricsDict, startDateTime, endDateTime)
    print('Program done')

if __name__ == "__main__":
    main()
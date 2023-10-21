import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from RevIn import RevIN
import ray
from ray import tune
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
import os

class nnWithReversible(nn.Module):
    def __init__(self, numDen=3, dropout=0.5, a=23, a1=23, a2=23, a3=23):
        super(nnWithReversible, self).__init__()
        self.revInLayer = RevIN(23)
        self.dense1 = nn.Linear(23, a)
        self.dense2 = nn.Linear(a, a1)
        self.dense3 = nn.Linear(a1, a2)
        self.dense4 = nn.Linear(a2, a3)
        self.a1ToRev = nn.Linear(a1, 23)
        self.a2ToRev = nn.Linear(a2, 23)
        self.a3ToRev = nn.Linear(a3, 23)
        self.denseOut = nn.Linear(23, 1)
        self.numDen = numDen
        self.dropout = dropout
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.double()

    def forward(self, x):
        # Use RevIn to normalize x
        x = self.revInLayer(x, 'norm')
        # Add the initial dense layer
        # x = F.relu(self.dense1(x))
        # for i in range(self.numDen):
        #     x = F.relu(self.dense1(x))
        #     x = nn.Dropout(p=self.dropout)(x)
        if (self.numDen == 2):
            x = F.relu(self.dense1(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense2(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.a1ToRev(x)) # convert for RevIn
        elif (self.numDen == 3):
            x = F.relu(self.dense1(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense2(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense3(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.a2ToRev(x)) # convert for RevIn
        elif (self.numDen == 4):
            x = F.relu(self.dense1(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense2(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense3(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.dense4(x))
            x = nn.Dropout(p=self.dropout)(x)
            x = F.relu(self.a3ToRev(x)) # convert for RevIn
        # Use RevIn to denormalize x
        x = self.revInLayer(x, 'denorm')
        # Add the final dense layer
        x = self.denseOut(x)
        # Return the output
        return x

def readEconData(filename):
    return pd.read_excel(filename)

def makeTrainTest(modelName, window, testWindow, secondTime=False): # Train test but with breaking up between pre-2020 and 2020->beyond
    # Read econ data
    # econData = readEconData("Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    econData = readEconData("C:/Users/zs811/OneDrive/Documents/CodingProjectsStuff/2023SummerResearch-Inflation/COVID-Inflation/Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    if modelName != "RNN":
        if not secondTime:
            # split into train/test 
            trainDf, testDf = econData.iloc[:window], econData.iloc[window:window+testWindow]
            # split into x and y
            # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
            # xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
        else:
            if window == 346: #First post2020 
                trainDf, testDf = econData.iloc[window-6:window], econData.iloc[window:window+testWindow]
                # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                # xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
            else:
                trainDf, testDf = econData.iloc[window-testWindow:window], econData.iloc[window:window+testWindow]
                # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                # xTest, yTest = testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
    else:
        # Specific train/test split for the RNN due to it needing to pull some values in the train window for the xTest
        if not secondTime:
            trainDf = econData.iloc[:window]
            # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
            # xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
            # yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
        else:
            if window == 346: #First post2020
                trainDf = econData.iloc[window-6-12:window]
                # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                # xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
                # yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
            else:
                trainDf = econData.iloc[window-testWindow-12:window]
                # xTrain, yTrain = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"]
                # xTest = econData.iloc[window-12:window+testWindow-1].loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"]
                # yTest = econData.iloc[window:window+testWindow].loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
    # return xTrain, yTrain, xTest, yTest
    return trainDf, testDf

def compileMetrics(metricsDict):
    # compile the metrics into two dataframes, the first for training metrics and the second for testing metrics
    trainingMetrics = pd.DataFrame(columns = ["cvMSE", "cvRMSE", "cvMAE","Train R^2", "Train Adjusted R^2", "Train MSE", "Train RMSE", "Train MAE", "Train Pearson's Correlation Coefficient"])
    testingMetrics = pd.DataFrame(columns=["Test R^2", "Test Adjusted R^2", "Test MSE", "Test RMSE", "Test MAE", "Test Pearson's Correlation Coefficient"])
    for key in metricsDict.keys():
        if "LR" in key or "RF" in key:
            trainingMetrics.loc[key] = metricsDict[key][:9]
            testingMetrics.loc[key] = metricsDict[key][9:]
        else:
            trainingMetrics.loc[key] = [0,0,0]+list(metricsDict[key][:6])
            testingMetrics.loc[key] = list(metricsDict[key][6:])
    return trainingMetrics, testingMetrics

def getModelMetrics(x, y, model, modelName, training=True):
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
    return r2, adjR2, mse, rmse, mae, corr
# https://mengliuz.medium.com/hyperparameter-tuning-for-deep-learning-models-with-the-ray-simple-pytorch-example-da7b17e3505
# EPOCH_SIZE = 512
# TEST_SIZE = 256
def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx * len(data) > EPOCH_SIZE:
        #     return
        data, target = data.to(device), target.to(device)
        pred = model(data)
        target = target.unsqueeze(1)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    # correct = 0
    # total = 0
    thisMSE = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # if batch_idx * len(data) > TEST_SIZE:
            #     break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            # _, predicted = torch.max(outputs.data, 1)
            # Get the MSE
                                 # predicted
            target = target.unsqueeze(1)
            thisMSE += F.mse_loss(outputs, target, reduction='mean').item()
    return thisMSE

# ref: https://gist.github.com/conormm/5b26a08029b900520bcd6fcd1f5712a0
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

def get_data_loaders(window, testWindow,batchSize, loadModel=False):
    trainDf, testDf = makeTrainTest("NN", window, testWindow, secondTime=loadModel)
    xTrain, yTrain, xTest, yTest = trainDf.loc[:, trainDf.columns != "AnnualizedMoM-CPI-Inflation"], trainDf.loc[:, trainDf.columns == "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns != "AnnualizedMoM-CPI-Inflation"], testDf.loc[:, testDf.columns == "AnnualizedMoM-CPI-Inflation"]
    # convert yTrain and yTest to numpy arrays for use in PrepareData
    yTrain, yTest = np.array([value[0] for value in yTrain.values.tolist()]), np.array([value[0] for value in yTest.values.tolist()])
    trainTorch, testTorch = PrepareData(X=xTrain, y=yTrain, scale_X=False), PrepareData(X=xTest, y=yTest, scale_X = False)
    # trainTorch, testTorch = DataFrame(X=trainDf.drop("AnnualizedMoM-CPI-Inflation", axis=1), y=trainDf["AnnualizedMoM-CPI-Inflation"]), DataFrame(X=testDf.drop("AnnualizedMoM-CPI-Inflation", axis=1), y=testDf["AnnualizedMoM-CPI-Inflation"])
    # print(trainTorch.df)
    # print(type(trainTorch))
    train_loader = DataLoader(trainTorch, batch_size=batchSize, shuffle=True)
    # print(type(train_loader))
    test_loader = DataLoader(testTorch, batch_size=batchSize, shuffle=True)
    return train_loader, test_loader

def train_mnist(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders(config["win"], config["testWin"],config["batch"], config["load"])
    model = nnWithReversible(numDen=config["numDen"], dropout=config["drop"], a=config["a"], a1=config["a1"], a2=config["a2"], a3=config["a3"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    while True:
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)
        # Set this to run Tune.
        # metrics = {"mean_loss": acc}
        # ray.train.report(metrics) # if you want to evaluate loss, change it to mean_loss and set the mode in tune.run to min instead
        tune.report(mean_loss = acc)


def trainEvalNN(window, testWindow, loadModel=False):
    if not loadModel:
        ray.init(num_cpus=8, num_gpus=4) # assign the total # of cpus and gpus, make sure you have ray.init in the beginning and ray.shutdown at the end
        sched = AsyncHyperBandScheduler()  # set a scheduler
        algo = HyperOptSearch() # if you want to use the Bayesian optimization, import BayesOptSearch instead
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
        # See the list of all search algorithms here: https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
        analysis = tune.run(
                train_mnist,   # the core training/testing of your model
                storage_path=os.getcwd(), # for saving the log files
                name="exp", # name for the result directory
                metric='mean_loss',
                mode='min',
                search_alg=algo,
                scheduler=sched,
                stop={
                        "mean_loss": 0.03,
                        "training_iteration": 100
                },
                resources_per_trial={
                        "cpu": 2,
                        "gpu": 1
                },
            num_samples=200, # num trials
            config = {"numDen": tune.choice([2,3,4]), 
                "drop": tune.choice([0.3, 0.5,0.7,0.85]),
                "batch": tune.choice([8,16,32]),
                "lr": tune.loguniform(1e-4, 1e-1),
                "a": tune.choice([23, 46, 69, 92]),
                "a1": tune.choice([23, 46, 69, 92]),
                "a2": tune.choice([23, 46, 69, 92]),
                "a3": tune.choice([23, 46, 69, 92]),
                "win": window,
                "testWin": testWindow,
                "load": loadModel
        })
        print("Best config is:", analysis.best_config)
        bestNumLayers, bestDrop, = analysis.best_config["numDen"], analysis.best_config["drop"]
        bestBatchSize, bestLR = analysis.best_config["batch"], analysis.best_config["lr"]
        bestA, bestA1, bestA2, bestA3 = analysis.best_config["a"], analysis.best_config["a1"], analysis.best_config["a2"], analysis.best_config["a3"]
        # save all the "best" variables to a file
        with open("Models/NNRevBestHyperparams.txt", "w") as f:
            f.write("Best number of dense layers: "+str(bestNumLayers)+"\n")
            f.write("Best dropout: "+str(bestDrop)+"\n")
            f.write("Best batch size: "+str(bestBatchSize)+"\n")
            f.write("Best learning rate: "+str(bestLR)+"\n")
            f.write("Best a: "+str(bestA)+"\n")
            f.write("Best a1: "+str(bestA1)+"\n")
            f.write("Best a2: "+str(bestA2)+"\n")
            f.write("Best a3: "+str(bestA3)+"\n")
        model = nnWithReversible(bestNumLayers, bestDrop, bestA, bestA1, bestA2, bestA3)
        train_loader, test_loader = get_data_loaders(window, testWindow, bestBatchSize, loadModel)
        optimizer = optim.SGD(model.parameters(), bestLR)
        trainDataset, testDataset = train_loader.dataset, test_loader.dataset
        xTrain, yTrain = trainDataset.X, trainDataset.y
        xTest, yTest = testDataset.X, testDataset.y
        errors = []
        bestError = 999999999
        earlyStopThresh = 4
        bestEpoch = -1
        for epoch in range(1, 100 + 1):
            train(model, optimizer, train_loader)
            epochError = test(model, test_loader)
            errors.append(epochError)
            print(f"Epoch {epoch}: Test MSE: {epochError}")
            if epochError < bestError:
                bestError = epochError
                bestEpoch = epoch
                r2, adjR2, mse, rmse, mae, corr = getModelMetrics(xTrain, yTrain, model, "NN", training=True)
                r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest = getModelMetrics(xTest, yTest, model, "NN", training=False)
                torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': analysis.best_config["batch"]},"Models/NNRevModel.pt")
            elif epoch - bestEpoch >= earlyStopThresh:
                print("Early stopping at epoch "+str(epoch))
                break
        ray.shutdown()
        # print the model weights
        # for param in model.parameters():
        #     print(param.data)
        # get xTrain and yTrain from the train_loader, and xTest and yTest from the test_loader
        
        '''
        # get the trained model's metrics
        r2, adjR2, mse, rmse, mae, corr = getModelMetrics(xTrain, yTrain, model, "NN", training=True)
        r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest = getModelMetrics(xTest, yTest, model, "NN", training=False)
        # save the trained model
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': analysis.best_config["batch"]},"Models/NNRevModel.pt")
        '''
        
        return r2, adjR2, mse, rmse, mae, corr, r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest
    else:
        # load the best hyperparameters from the file
        with open("Models/NNRevBestHyperparams.txt", "r") as f:
            bestNumLayers, bestDrop = int(f.readline().split(": ")[1]), float(f.readline().split(": ")[1])
            bestBatchSize, bestLR = int(f.readline().split(": ")[1]), float(f.readline().split(": ")[1])
            bestA, bestA1, bestA2, bestA3 = int(f.readline().split(": ")[1]), int(f.readline().split(": ")[1]), int(f.readline().split(": ")[1]), int(f.readline().split(": ")[1])
        # load the model and optimizer from the file
        model = nnWithReversible(bestNumLayers, bestDrop, bestA, bestA1, bestA2, bestA3)
        optimizer = optim.SGD(model.parameters(), lr=bestLR)
        checkpoint = torch.load("Models/NNRevModel.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch_size = checkpoint['batch_size']
        # get xTrain and yTrain from the train_loader, and xTest and yTest from the test_loader
        train_loader, test_loader = get_data_loaders(window, testWindow, batch_size, loadModel)
        trainDataset, testDataset = train_loader.dataset, test_loader.dataset
        xTrain, yTrain = trainDataset.X, trainDataset.y
        xTest, yTest = testDataset.X, testDataset.y
        # train the model for 100 epochs
        errors = []
        bestError = 999999999
        earlyStopThresh = 5
        bestEpoch = -1
        r2, adjR2, mse, rmse, mae, corr, r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest = -999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999
        for epoch in range(1, 100 + 1):
            train(model, optimizer, train_loader)
            epochError = test(model, test_loader)
            errors.append(epochError)
            print(f"Epoch {epoch}: Test MSE: {epochError}")
            if epochError < bestError:
                bestError = epochError
                bestEpoch = epoch
                r2, adjR2, mse, rmse, mae, corr = getModelMetrics(xTrain, yTrain, model, "NN", training=True)
                r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest = getModelMetrics(xTest, yTest, model, "NN", training=False)
                torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': batch_size},"Models/NNRevModel.pt")
            elif epoch - bestEpoch >= earlyStopThresh:
                print("Early stopping at epoch "+str(epoch))
                break
        '''
        # print the model weights
        # for param in model.parameters():
        #     print(param.data)
        # get the trained model's metrics
        r2, adjR2, mse, rmse, mae, corr = getModelMetrics(xTrain, yTrain, model, "NN", training=True)
        r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest = getModelMetrics(xTest, yTest, model, "NN", training=False)
        # save the trained model
        torch.save({'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'batch_size': batch_size},"Models/NNRevModel.pt")
        '''
        return r2, adjR2, mse, rmse, mae, corr, r2Test, adjR2Test, mseTest, rmseTest, maeTest, corrTest

def main():
    loadModel = False
    pre2020trainWindowSize, pre2020testWindowSize = [200 + 10*i for i in range(14)], 10
    pre2020trainMetrics, pre2020testMetrics = [], []
    transTrainWindowSize, transTestWindowSize = [335], 5
    transTrainMetrics, transTestMetrics = [], []
    post2020trainWindowSize, post2020testWindowSize = [346 + 2*i for i in range(16)], 2
    post2020trainMetrics, post2020testMetrics = [], []
    for count, window in enumerate(pre2020trainWindowSize):
        if count == 0:
            metricsDict = {"NN"+str(window): trainEvalNN(window, pre2020testWindowSize, loadModel)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
        else:
            metricsDict = {"NN"+str(window): trainEvalNN(window, pre2020testWindowSize, loadModel=True)}
            train, test = compileMetrics(metricsDict)
            pre2020trainMetrics.append(train)
            pre2020testMetrics.append(test)
    for window in transTrainWindowSize:
        metricsDict = {"NN"+str(window): trainEvalNN(window, transTestWindowSize, loadModel=True) }
        train, test = compileMetrics(metricsDict)
        transTrainMetrics.append(train)
        transTestMetrics.append(test)
    for count, window in enumerate(post2020trainWindowSize):
        metricsDict = {"NN"+str(window): trainEvalNN(window, post2020testWindowSize, loadModel=True)}
        train, test = compileMetrics(metricsDict)
        post2020trainMetrics.append(train)
        post2020testMetrics.append(test)
    pre2020train, pre2020test = pd.concat(pre2020trainMetrics), pd.concat(pre2020testMetrics)
    transtrain, transtest = pd.concat(transTrainMetrics), pd.concat(transTestMetrics)
    post2020train, post2020test = pd.concat(post2020trainMetrics), pd.concat(post2020testMetrics)
    # add a row at the end which is the average of each column for rows containing "NN"
    pre2020train.loc['Pre2020NNavg'] = pre2020train.loc[pre2020train.index.str.contains("NN")].mean()
    pre2020test.loc['Pre2020NNavg'] = pre2020test.loc[pre2020test.index.str.contains("NN")].mean()
    transtrain.loc['TransNNavg'] = transtrain.loc[transtrain.index.str.contains("NN")].mean()
    transtest.loc['TransNNavg'] = transtest.loc[transtest.index.str.contains("NN")].mean()
    post2020train.loc['Post2020NNavg'] = post2020train.loc[post2020train.index.str.contains("NN")].mean()
    post2020test.loc['Post2020NNavg'] = post2020test.loc[post2020test.index.str.contains("NN")].mean()
    # combine the train and test dataframes into one
    train = pd.concat([pre2020train,transtrain, post2020train])
    test = pd.concat([pre2020test, transtest, post2020test])
    # move rows that contain "avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("avg")],train[train.index.str.contains("avg")]])
    test = pd.concat([test[~test.index.str.contains("avg")],test[test.index.str.contains("avg")]])
    # START OF TOTAL AVERAGE CODE
    # Repeat the above steps for the NN model
    temp = train.loc[train.index.str.contains("Pre2020NNavg")]*10
    temp2 = train.loc[train.index.str.contains("TransNNavg")]*5
    temp3 = train.loc[train.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    train = pd.concat([train, temp])
    # train.rename(index={train.index[-1]:'TotalNNAvg'}, inplace=True)
    temp = test.loc[test.index.str.contains("Pre2020NNavg")]*10
    temp2 = test.loc[test.index.str.contains("TransNNavg")]*5
    temp3 = test.loc[test.index.str.contains("Post2020NNavg")]*2
    for i in range(len(temp2.columns)):
        temp.iloc[0,i] = temp.iloc[0,i] + temp2.iloc[0,i] + temp3.iloc[0,i]
    temp = temp/17
    # Rename the index of temp to TotalNNAvg
    temp.rename(index={temp.index[0]:'TotalNNAvg'}, inplace=True)
    test = pd.concat([test, temp])
    # test.rename(index={test.index[-1]:'TotalNNAvg'}, inplace=True)
    # move the rows that contain "Avg" to the bottom
    train = pd.concat([train[~train.index.str.contains("avg")],train[train.index.str.contains("avg")]])
    test = pd.concat([test[~test.index.str.contains("avg")],test[test.index.str.contains("avg")]])
    # save the dataframes to excel files
    train.to_excel("Metrics/newNNFullHyperTrain.xlsx")
    test.to_excel("Metrics/newNNFullHyperTest.xlsx")
    print("Program Done")

if __name__ == "__main__":
    main()
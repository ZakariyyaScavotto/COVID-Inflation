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
from tensorflow.keras.models import load_model

def readEconData(filename):
    df = pd.read_excel(filename)
    # Last number in HousingPriceInd missing so dropping that row
    df.dropna(subset=["HPI%Change"], inplace=True)
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
    # reference: https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    explainer = shap.explainers.Linear(myLR, xTrain)
    shap_values = explainer.shap_values(xTrain)
    shap.summary_plot(shap_values, xTrain, feature_names=xTrain.columns, plot_type="bar", show=False)
    plt.title("Feature Importance for Linear Regression")
    plt.gcf().canvas.manager.set_window_title("Feature Importance for Linear Regression")
    plt.gcf().set_size_inches(10,6)
    plt.show()
    return xTest, yTest, myLR

def evaluateLR(xTest, yTest, myLR):
    # Evaluate the LR by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myLR.predict(xTest)
    print("LR MSE: ", mean_squared_error(yTest, predictions))
    print("LR R^2: ", r2_score(yTest, predictions))
    print("LR Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))

def trainRF(econData):
    myRF = RandomForestRegressor()
    xTrain, yTrain, xTest, yTest = makeTrainTest(econData)
    myRF.fit(xTrain, yTrain.values.ravel())
    print("Finished training RF")
    # reference: https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    explainer = shap.TreeExplainer(myRF)
    shap_values = explainer.shap_values(xTrain)
    shap.summary_plot(shap_values, xTrain, feature_names=xTrain.columns, plot_type="bar", show=False)
    plt.title("Feature Importance for Random Forest")
    plt.gcf().canvas.manager.set_window_title("Feature Importance for Random Forest")
    plt.gcf().set_size_inches(10,6)
    plt.show()
    return xTest, yTest, myRF

def evaluateRF(xTest, yTest, myRF):
    # Evaluate the RF by getting predictions on xTest, then calculating the MSE and R^2
    predictions = myRF.predict(xTest)
    print("RF MSE: ", mean_squared_error(yTest, predictions))
    print("RF R^2: ", r2_score(yTest, predictions))
    print("RF Adjusted R^2: ", 1 - (1-r2_score(yTest, predictions)) * (len(yTest)-1)/(len(yTest)-xTest.shape[1]-1))

def trainNN(econData):
    myNN = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(300, activation='relu', input_shape=[15]),
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
    xTrain, yTrain, xTest, yTest = makeTrainTest(econData)
    history = myNN.fit(
    xTrain, yTrain,
    validation_data=(xTest, yTest),
    batch_size=100,
    epochs=100,
    verbose=1, # decided to make verbose to follow the training, feel free to set to 0
    callbacks=[es]
    )
    print("Finished training NN")
    return xTest, yTest, myNN

def evaluateNN(xTest, yTest, myNN):
    test_loss = myNN.evaluate(xTest, yTest, verbose=0)
    # print(myNN.metrics_names)
    print("NN Loss: ", test_loss)


def main():
    # Read in the full econ data file
    econData = readEconData("Data\EconomicData\ALLECONDATA.xlsx")
    correlationMatrix = econData.corr()
    correlationMatrix.to_excel("Data\CorrelationMatrix.xlsx")
    # Try basic LR on the econ data
    xTest, yTest, firstLR = trainLR(econData)
    evaluateLR(xTest, yTest, firstLR)
    # Try basic RF on the econ data
    xTest, yTest, firstRF = trainRF(econData)
    evaluateRF(xTest, yTest, firstRF)
    # Try basic NN on the econ data
    xTest, yTest, firstNN = trainNN(econData)
    evaluateNN(xTest, yTest, firstNN)
    print("Program Done")

if __name__ == "__main__":
    main()

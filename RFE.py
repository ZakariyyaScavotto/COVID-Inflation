import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

def readEconData(filename):
    return pd.read_excel(filename)

def makeXY(): 
    # Read econ data
    # econData = readEconData("Data\ConstructedDataframes\AllEcon1990AndCOVIDWithLags.xlsx")
    econData = readEconData("Data\ConstructedDataframes\AutoregressiveAllLags.xlsx")
    # drop the date column
    econData.drop("Date", axis=1, inplace=True)
    # scale the data using StandardScaler
    scaler = StandardScaler()
    econData = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
    # Split into x and y
    x, y = econData.loc[:, econData.columns != "AnnualizedMoM-CPI-Inflation"], econData.loc[:, econData.columns == "AnnualizedMoM-CPI-Inflation"]
    return x, y

# https://machinelearningmastery.com/rfe-feature-selection-in-python/
def recursiveFeatureElimination(modelName, minFeatures):
    x, y = makeXY()
    if modelName == "LR":
        print("Doing recursive feature elimination for Linear Regression model...")
        myLR = LinearRegression()
        rfecv = RFECV(estimator=myLR, scoring="neg_mean_squared_error", min_features_to_select=minFeatures)
        rfecv.fit(x, y)
        for i in range(x.shape[1]):
            print('Feature: %s, Selected: %s, Rank: %.3f' % (x.columns.values[i], rfecv.support_[i], rfecv.ranking_[i]))
        supportedFeatures = x.columns[rfecv.support_]
        print("Supported features: ", supportedFeatures)
        print("Done with recursive feature elimination for Linear Regression model")
        return supportedFeatures
    elif modelName == "RF":
        print("Doing recursive feature elimination for Random Forest model...")
        myRF = RandomForestRegressor()
        rfecv = RFECV(estimator=myRF, scoring="neg_mean_squared_error", min_features_to_select=minFeatures)
        rfecv.fit(x, y.values.ravel())
        for i in range(x.shape[1]):
            print('Feature: %s, Selected: %s, Rank: %.3f' % (x.columns.values[i], rfecv.support_[i], rfecv.ranking_[i]))
        supportedFeatures = x.columns[rfecv.support_]
        print("Supported features: ", supportedFeatures)
        print("Done with recursive feature elimination for Random Forest model")
        return supportedFeatures
    else:
        print("Invalid model name. Please enter either LR or RF")
        return None

def main():
    x,y = makeXY()
    LR = pd.DataFrame(columns=x.columns)
    RF = pd.DataFrame(columns=x.columns)
    for minFeature in range(1, 11):
        models = ["LR", "RF"]
        for model in models:
            supported = recursiveFeatureElimination(model, minFeature)
            if model == "LR":
                # Have the supported features for LR be 1 if supported, 0 if not
                LR.loc[minFeature] = [1 if feature in supported else 0 for feature in x.columns]
            elif model == "RF":
                # Have the supported features for RF be 1 if supported, 0 if not
                RF.loc[minFeature] = [1 if feature in supported else 0 for feature in x.columns]
    LR.to_excel("RFE\\AutoRegressiveLR.xlsx")
    RF.to_excel("RFE\\AutoRegressiveRF.xlsx")
    print("Program Done")

if __name__ == "__main__":
    main()
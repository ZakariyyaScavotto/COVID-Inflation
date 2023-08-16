import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readMetricFile(filename, train=False):
    df = pd.read_excel(filename)
    # set the index to be the first columnn
    df.set_index(df.columns[0], inplace=True)
    if train:
        if 'Multi' not in filename:
            df.drop(columns=['cvMSE', 'cvRMSE','cvMAE',	'Train R^2','Train Adjusted R^2',"Train Pearson's Correlation Coefficient"], inplace=True)
        else:
            df.drop(columns=['Train R^2','Train Adjusted R^2',"Train Pearson's Correlation Coefficient"], inplace=True)
    else:
        df.drop(columns=["Test R^2",'Test Adjusted R^2',"Test Pearson's Correlation Coefficient"], inplace=True)
    return df

def getDates():
    df = pd.read_excel("Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx")
    dates = df['Date']
    keyIndices = [200 + 10*i for i in range(15)]
    keyIndices.extend([340 + 2*i for i in range(19)])
    dates = dates[keyIndices]
    return dates

def separateAvg(df):
    avgs = pd.concat([df.loc[df.index.str.contains("avg")],df.loc[df.index.str.contains("Avg")]])
    nonAvgs = df.drop(avgs.index)
    return nonAvgs, avgs

def generateNonAvgPlots(df, filename):
    # Create a subplot of one row and three columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Set title for the figure
    if 'Train' in filename:
        if 'PossibleRemove' in filename:
            fig.suptitle('Train Metrics After Possible Removal')
        else:
            fig.suptitle('Train Metrics')
    else:
        if 'PossibleRemove' in filename:
            fig.suptitle('Test Metrics After Possible Removal')
        else:
            fig.suptitle('Test Metrics')
    modelNames = ['RF', 'NN', 'RNN']
    for model in modelNames:
        modelData = df.loc[df.index.str.contains(model)]
        if model == 'NN':
            # drop all indices that have 'RNN' in them
            modelData = modelData.drop(modelData.index[modelData.index.str.contains('RNN')])
        # Rest the index to default integer index
        modelData.reset_index(inplace=True, drop=True)
        if 'Test' in filename:
            axs[0].plot(modelData['Test MSE'], label=model)
            axs[1].plot(modelData['Test RMSE'], label=model)
            axs[2].plot(modelData['Test MAE'], label=model)
        else:
            axs[0].plot(modelData['Train MSE'], label=model)
            axs[1].plot(modelData['Train RMSE'], label=model)
            axs[2].plot(modelData['Train MAE'], label=model)
    # set the x axis of each subplot to be the dates
    dates = getDates()
    # Change the dates to only being the month and year
    dates = [date.strftime("%b-%Y") for date in dates]
    axs[0].set_xticks(np.arange(0, len(dates), 4))
    axs[0].set_xticklabels(dates[::4], rotation=45)
    axs[1].set_xticks(np.arange(0, len(dates), 4))
    axs[1].set_xticklabels(dates[::4], rotation=45)
    axs[2].set_xticks(np.arange(0, len(dates), 4))
    axs[2].set_xticklabels(dates[::4], rotation=45)
    axs[0].tick_params(axis='x', labelsize=8)
    axs[1].tick_params(axis='x', labelsize=8)
    axs[2].tick_params(axis='x', labelsize=8)
    if 'Test' in filename:
        axs[0].set_title('Test MSE')
        axs[1].set_title('Test RMSE')
        axs[2].set_title('Test MAE')
    else:
        axs[0].set_title('Train MSE')
        axs[1].set_title('Train RMSE')
        axs[2].set_title('Train MAE')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.savefig('Metrics/PlotsNoLRNoEnsemble/'+filename.replace('.xlsx', '.png').replace('Metrics/',''))

def generateAvgPlots(df, filename):
    # create a subplot of barplots three rows and three columns
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # Set title for the figure
    if 'Train' in filename:
        if 'PossibleRemove' in filename:
            fig.suptitle('Avg Train Metrics After Possible Removal')
        else:
            fig.suptitle('Avg Train Metrics')
    else:
        if 'PossibleRemove' in filename:
            fig.suptitle('Avg Test Metrics After Possible Removal')
        else:
            fig.suptitle('Avg Test Metrics')
    time = ['Pre', 'Post', 'Total']
    modelNames = ['RF', 'NN', 'RNN']
    for time in time:
        timeDf = df.loc[df.index.str.contains(time)]
        for model in modelNames:
            modelData = timeDf.loc[timeDf.index.str.contains(model)]
            # the first row of the subplot is for the Pre time, the second row is for the Post time, and the third row is for the Total time
            if time == 'Pre':
                row = 0
            elif time == 'Post':
                row = 1
            else:
                row = 2
            # do barplot of the values
            if 'Test' in filename:
                axs[row][0].bar(x=model, height=modelData['Test MSE'], label=model)
                axs[row][1].bar(x=model,height=modelData['Test RMSE'], label=model)
                axs[row][2].bar(x=model,height=modelData['Test MAE'], label=model)
            else:
                axs[row][0].bar(x=model,height=modelData['Train MSE'], label=model)
                axs[row][1].bar(x=model,height=modelData['Train RMSE'], label=model)
                axs[row][2].bar(x=model,height=modelData['Train MAE'], label=model)
    # set the title of all 9 subplots
    if 'Test' in filename:
        axs[0][0].set_title('Pre2020 Test MSE')
        axs[0][1].set_title('Pre2020Test RMSE')
        axs[0][2].set_title('Pre2020 Test MAE')
        axs[1][0].set_title('Post2020 Test MSE')
        axs[1][1].set_title('Post2020 Test RMSE')
        axs[1][2].set_title('Post2020 Test MAE')
        axs[2][0].set_title('Total Test MSE')
        axs[2][1].set_title('Total Test RMSE')
        axs[2][2].set_title('Total Test MAE')
    else:
        axs[0][0].set_title('Pre2020 Train MSE')
        axs[0][1].set_title('Pre2020 Train RMSE')
        axs[0][2].set_title('Pre2020 Train MAE')
        axs[1][0].set_title('Post2020 Train MSE')
        axs[1][1].set_title('Post2020 Train RMSE')
        axs[1][2].set_title('Post2020 Train MAE')
        axs[2][0].set_title('Total Train MSE')
        axs[2][1].set_title('Total Train RMSE')
        axs[2][2].set_title('Total Train MAE')
    # save the figure
    fig.savefig('Metrics/PlotsNoLRNoEnsemble/'+filename.replace('.xlsx', 'Avg.png').replace('Metrics/',''))

def mainPlotting():
    filesToRead = ['Metrics/rollingTestMetrics.xlsx', 'Metrics/rollingTrainMetrics.xlsx', 'PossibleRemoveMetrics/rollingTestMetrics.xlsx', 'PossibleRemoveMetrics/rollingTrainMetrics.xlsx',
                   'MultioutputMetrics/rollingTestMetrics.xlsx', 'MultioutputMetrics/rollingTrainMetrics.xlsx',
                   'Brent3COVIDMetrics/rollingTestMetrics.xlsx', 'Brent3COVIDMetrics/rollingTrainMetrics.xlsx']
    for metricFile in filesToRead:
        if 'Train' in metricFile:
            train = True
        else:
            train = False
        nonAvgs, avgs = separateAvg(readMetricFile(metricFile, train))
        generateNonAvgPlots(nonAvgs, metricFile)
        generateAvgPlots(avgs, metricFile)
    print('Plotting Program (No LR No Ensemble) Done')

if __name__ == '__main__':
    mainPlotting()
import pandas as pd

def generateFileTables():
    filesToRead = ['Metrics/rollingTestMetrics.xlsx', 'PossibleRemoveMetrics/rollingTestMetrics.xlsx','Brent3COVIDMetrics/rollingTestMetrics.xlsx']
    for file in filesToRead:
        df = pd.read_excel(file)
        # set the index of the df to be the first column
        df.set_index(df.columns[0], inplace=True)
        # drop the first column
        df.drop(df.columns[0], axis=1, inplace=True)
        # drop every row that doesn't contain "avg" or "Avg"
        df.drop(df.index[~df.index.str.contains("avg|Avg")], inplace=True)
        df.to_excel(f"Metrics/fileTables/{file.split('/')[0]}.xlsx")
    print("Finished generating file tables")

def generateModelTables():
    filesToRead = ['Metrics/rollingTestMetrics.xlsx', 'PossibleRemoveMetrics/rollingTestMetrics.xlsx','Brent3COVIDMetrics/rollingTestMetrics.xlsx']
    models = ['LR', 'RF', 'NN', 'RNN', 'Ensemble']
    timePeriod = ['Pre2020', 'Post2020', 'Total']
    for model in models:
        for time in timePeriod:
            modelDf = pd.DataFrame()
            for file in filesToRead:
                df = pd.read_excel(file)
                # set the index of the df to be the first column
                df.set_index(df.columns[0], inplace=True)
                # drop the first column
                df.drop(df.columns[0], axis=1, inplace=True)
                # find the row in index with the model and time period
                modelRows = df.loc[df.index.str.contains(model)]
                if model == 'NN':
                    # drop all indices that have 'RNN' in them
                    modelRows = modelRows.drop(modelRows.index[modelRows.index.str.contains('RNN')])
                rowToAdd = modelRows.loc[modelRows.index.str.contains(time)]
                rowToAdd.rename(index={rowToAdd.index[0]: file.split('/')[0]+rowToAdd.index[0]}, inplace=True)
                modelDf = modelDf.append(rowToAdd)
            modelDf.to_excel(f"Metrics/modelTables/{model}/{time}.xlsx")
    print("Finished generating metric tables by model")

generateModelTables()
generateFileTables()
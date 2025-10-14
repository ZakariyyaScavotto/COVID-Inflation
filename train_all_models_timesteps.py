#!/usr/bin/env python3
"""
Comprehensive Multi-Model Training Script for COVID-Inflation Analysis

Trains LR/DynLR, RF, NN/DynNN, RNN, LSTM, GRU (and Ensemble if available) across
multiple datasets and lags, with RNN timesteps. Saves detailed results and timing
breakdowns. Matches post-2020 (secondTime=True) methodology used elsewhere.

Outputs:
- training_results_detailed.csv
- training_results_summary.txt
- training_timing_results.csv
"""

import os
import sys
import time
import pickle
import warnings
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional DL stack
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except Exception as e:
    print(f"⚠️ TensorFlow/Keras not available: {e}")
    HAS_TF = False

# Required: scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")

# Optional: custom ensemble
HAS_ENSEMBLE = False
try:
    sys.path.append('/home/ubuntu/COVID-Inflation')
    from myEnsemble import myEnsembleModel
    HAS_ENSEMBLE = True
except Exception as e:
    print(f"⚠️ Custom Ensemble not available: {e}")


class ComprehensiveModelTrainer:
    def __init__(self):
        self.results = []
        self.timing = {}
        self.scaler = StandardScaler()

    # ---------------- Data loading ---------------- #
    def load_data(self, dataset_type: str) -> pd.DataFrame:
        if dataset_type == 'regular':
            path = 'Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx'
        elif dataset_type == 'interp':
            path = 'Data/ConstructedDataframes/INTERPAllEcon1990AndCOVIDWithLags.xlsx'
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        print(f"Loading {dataset_type} dataset from {path}")
        df = pd.read_excel(path)

        if dataset_type == 'interp':
            before = len(df)
            df = df.iloc[::4].reset_index(drop=True)
            after = len(df)
            print(f"🔧 INTERP synthetic removal: {before} -> {after}")

        return df

    # ------------- Data preparation ------------- #
    def prepare_data_for_model(self, df: pd.DataFrame, model_type: str, lag: int, timestep: int = None):
        # Target column is the current inflation we're trying to predict
        inflation_col = "AnnualizedMoM-CPI-Inflation"
        if inflation_col not in df.columns:
            raise ValueError(f"Missing column: {inflation_col}")

        # Drop Date column if present, then dropna
        econ = df.drop('Date', axis=1, errors='ignore').dropna().reset_index(drop=True)
        econ_scaled = pd.DataFrame(self.scaler.fit_transform(econ), columns=econ.columns)

        # Reference post-2020 evaluation params
        window = 346
        testWindow = 2
        secondTime = True

        if model_type in ['RNN', 'LSTM', 'GRU'] and timestep:
            # RNN family: need 6 + timestep training rows when secondTime
            if secondTime and window == 346:
                trainDf = econ_scaled.iloc[window - 6 - timestep:window]
            else:
                trainDf = econ_scaled.iloc[window - testWindow - timestep:window]

            xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
            yTrain = trainDf.loc[:, trainDf.columns == inflation_col]

            test_start = window - timestep
            test_end = window + testWindow - 1
            xTest = econ_scaled.iloc[test_start:test_end].loc[:, econ_scaled.columns != inflation_col]
            yTest = econ_scaled.iloc[window:window + testWindow].loc[:, econ_scaled.columns == inflation_col]

            X_train = xTrain.values
            X_test = xTest.values
            y_train = yTrain.values.ravel()
            y_test = yTest.values.ravel()

            X_train_seq = self.make_sequences(X_train, timestep)
            X_test_seq = self.make_sequences(X_test, timestep)
            if X_train_seq is None or X_test_seq is None:
                return None, None, None, None
            return X_train_seq, X_test_seq, y_train[:len(X_train_seq)], y_test[:len(X_test_seq)]

        # Non-RNN models
        if secondTime and window == 346:
            trainDf = econ_scaled.iloc[window - 6:window]
        else:
            trainDf = econ_scaled.iloc[:window]

        xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
        yTrain = trainDf.loc[:, trainDf.columns == inflation_col]
        xTest = econ_scaled.iloc[window:window + testWindow].loc[:, econ_scaled.columns != inflation_col]
        yTest = econ_scaled.iloc[window:window + testWindow].loc[:, econ_scaled.columns == inflation_col]

        return xTrain.values, xTest.values, yTrain.values.ravel(), yTest.values.ravel()

    @staticmethod
    def make_sequences(X: np.ndarray, timestep: int):
        n = X.shape[0]
        if n < timestep:
            return None
        seq = []
        for i in range(n - timestep + 1):
            seq.append(X[i:i + timestep])
        return np.asarray(seq)

    # ----------------- Trainers ----------------- #
    def train_lr(self, X, y):
        m = LinearRegression()
        m.fit(X, y)
        return m

    def train_rf(self, X, y):
        m = RandomForestRegressor(n_estimators=200, random_state=42)
        m.fit(X, y)
        return m

    def train_nn(self, X, y):
        if not HAS_TF:
            raise RuntimeError("TensorFlow unavailable for NN training")
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(1e-3), loss='mse')
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[es])
        return model

    def train_rnn_family(self, X_seq, y, rnn_type: str):
        if not HAS_TF:
            raise RuntimeError("TensorFlow unavailable for RNN training")
        units = 50
        model = Sequential()
        if rnn_type == 'RNN':
            model.add(SimpleRNN(units, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        elif rnn_type == 'LSTM':
            model.add(LSTM(units, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        elif rnn_type == 'GRU':
            model.add(GRU(units, input_shape=(X_seq.shape[1], X_seq.shape[2])))
        else:
            raise ValueError(f"Unknown rnn_type {rnn_type}")
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(1e-3), loss='mse')
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_seq, y, epochs=100, batch_size=32, verbose=0, callbacks=[es])
        return model

    def train_ensemble(self, X, y):
        if not HAS_ENSEMBLE:
            return None
        try:
            m = myEnsembleModel()
            m.fit(X, y)
            return m
        except Exception as e:
            print(f"⚠️ Ensemble training failed: {e}")
            return None

    # ---------------- Evaluation ---------------- #
    @staticmethod
    def evaluate(model, X_train, y_train, X_test, y_test, is_dl: bool):
        if is_dl:
            ytr = model.predict(X_train, verbose=0).ravel()
            yte = model.predict(X_test, verbose=0).ravel()
        else:
            ytr = model.predict(X_train)
            yte = model.predict(X_test)
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, ytr)))
        tr_mae = float(mean_absolute_error(y_train, ytr))
        te_rmse = float(np.sqrt(mean_squared_error(y_test, yte)))
        te_mae = float(mean_absolute_error(y_test, yte))
        return tr_rmse, tr_mae, te_rmse, te_mae

    @staticmethod
    def save_model_artifact(model, name: str, is_dl: bool) -> str:
        if is_dl:
            fname = f"{name}.h5"
            model.save(fname)
        else:
            fname = f"{name}.pickle"
            with open(fname, 'wb') as f:
                pickle.dump(model, f)
        return fname

    # ---------------- Orchestration -------------- #
    def train_all(self, models=None, dataset_types=("regular", "interp"), lags=(1, 3, 6, 12), timesteps=(6, 12, 18)):
        if models is None:
            models = ['LR', 'DynLR', 'RF', 'NN', 'DynNN', 'Ensemble', 'RNN', 'LSTM', 'GRU']
            
        non_rnn = [m for m in models if m in ['LR', 'DynLR', 'RF', 'NN', 'DynNN', 'Ensemble']]
        rnn_models = [m for m in models if m in ['RNN', 'LSTM', 'GRU']]

        total = len(dataset_types) * len(lags) * len(non_rnn) + len(dataset_types) * len(lags) * len(rnn_models) * len(timesteps)
        count = 0

        for dset in dataset_types:
            print(f"\n📊 Dataset: {dset}")
            df = self.load_data(dset)
            for lag in lags:
                print(f"\n  🎯 Lag {lag}")

                # Non-RNNs
                for m in non_rnn:
                    count += 1
                    print("\n" + "="*50)
                    print(f"🔍 {m} | {dset} | Lag {lag}")
                    print("="*50)
                    try:
                        t0 = time.time()
                        Xtr, Xte, ytr, yte = self.prepare_data_for_model(df, m, lag)
                        if Xtr is None:
                            raise RuntimeError("Preparation failed")

                        if m in ['LR', 'DynLR']:
                            model = self.train_lr(Xtr, ytr)
                            is_dl = False
                        elif m == 'RF':
                            model = self.train_rf(Xtr, ytr)
                            is_dl = False
                        elif m in ['NN', 'DynNN']:
                            model = self.train_nn(Xtr, ytr)
                            is_dl = True
                        elif m == 'Ensemble':
                            model = self.train_ensemble(Xtr, ytr)
                            if model is None:
                                print("❌ Ensemble unavailable; skipping")
                                continue
                            is_dl = False
                        else:
                            raise ValueError(m)

                        tr, ta, er, ea = self.evaluate(model, Xtr, ytr, Xte, yte, is_dl)
                        name = f"{m}Model_lag{lag}"
                        artifact = self.save_model_artifact(model, name, is_dl)
                        dt = time.time() - t0
                        self.timing[m] = self.timing.get(m, 0.0) + dt

                        self.results.append({
                            'Dataset': dset,
                            'Prediction_Lag': lag,
                            'Model': m,
                            'Timestep': 'N/A',
                            'Train_RMSE': tr,
                            'Train_MAE': ta,
                            'Test_RMSE': er,
                            'Test_MAE': ea,
                            'Training_Time_Seconds': dt,
                            'Model_File': artifact,
                            'Data_Shape_Train': str(Xtr.shape),
                            'Data_Shape_Test': str(Xte.shape)
                        })
                        print("✅ Evaluation complete:")
                        print(f"   Train: RMSE={tr:.4f}, MAE={ta:.4f}")
                        print(f"   Test:  RMSE={er:.4f}, MAE={ea:.4f}")
                        print(f"Progress: {count}/{total} ({count/total*100:.1f}%)")
                    except Exception as e:
                        print(f"❌ Error training {m}: {e}")

                # RNN-family
                for m in rnn_models:
                    for tstep in timesteps:
                        count += 1
                        print("\n" + "="*50)
                        print(f"🔍 {m} | {dset} | Lag {lag} | t{tstep}")
                        print("="*50)
                        try:
                            t0 = time.time()
                            Xtr, Xte, ytr, yte = self.prepare_data_for_model(df, m, lag, tstep)
                            if Xtr is None:
                                print(f"❌ Cannot create sequences for timestep {tstep}")
                                continue
                            model = self.train_rnn_family(Xtr, ytr, m)
                            tr, ta, er, ea = self.evaluate(model, Xtr, ytr, Xte, yte, True)
                            name = f"{m}Model_lag{lag}_t{tstep}"
                            artifact = self.save_model_artifact(model, name, True)
                            dt = time.time() - t0
                            self.timing[m] = self.timing.get(m, 0.0) + dt

                            self.results.append({
                                'Dataset': dset,
                                'Prediction_Lag': lag,
                                'Model': m,
                                'Timestep': tstep,
                                'Train_RMSE': tr,
                                'Train_MAE': ta,
                                'Test_RMSE': er,
                                'Test_MAE': ea,
                                'Training_Time_Seconds': dt,
                                'Model_File': artifact,
                                'Data_Shape_Train': str(Xtr.shape),
                                'Data_Shape_Test': str(Xte.shape)
                            })
                            print("✅ Evaluation complete:")
                            print(f"   Train: RMSE={tr:.4f}, MAE={ta:.4f}")
                            print(f"   Test:  RMSE={er:.4f}, MAE={ea:.4f}")
                            print(f"Progress: {count}/{total} ({count/total*100:.1f}%)")
                        except Exception as e:
                            print(f"❌ Error training {m} t{tstep}: {e}")

    def save_results(self, save_csv=True):
        if save_csv:
            # Detailed results with original column names
            pd.DataFrame(self.results).to_csv('training_results_detailed.csv', index=False)
            print("Detailed results saved to: training_results_detailed.csv")

            # Timing breakdown CSV
            pd.DataFrame([
                {'Model': k, 'Time_Seconds': v, 'Time_Formatted': str(timedelta(seconds=int(v)))}
                for k, v in self.timing.items()
            ]).to_csv('training_timing_results.csv', index=False)
            print("Timing results saved to: training_timing_results.csv")

        # Always save text summary
        lines = [
            "========================================",
            "TIMING BREAKDOWN:",
            "========================================",
        ]
        total_time = sum(self.timing.values())
        for k in sorted(self.timing.keys()):
            t = self.timing[k]
            lines.append(f"{k}: {str(timedelta(seconds=int(t)))} ({t:.1f}s)")
        lines.append(f"Total: {str(timedelta(seconds=int(total_time)))} ({total_time:.1f}s)")

        with open('training_results_summary.txt', 'w') as f:
            f.write("\n".join(lines))

        # Echo to console (so it appears in tmux capture)
        print("\n" + "\n".join(lines))
        print("Results summary saved to: training_results_summary.txt")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Comprehensive Model Training for COVID-Inflation Analysis')
    
    parser.add_argument('--models', nargs='+', 
                        choices=['LR', 'DynLR', 'RF', 'NN', 'DynNN', 'Ensemble', 'RNN', 'LSTM', 'GRU'],
                        default=['LR', 'DynLR', 'RF', 'NN', 'DynNN', 'Ensemble', 'RNN', 'LSTM', 'GRU'],
                        help='Models to train (default: all models)')
    
    parser.add_argument('--prediction-lags', nargs='+', type=int,
                        default=[1, 3, 6, 12],
                        help='Prediction lags to use (default: 1 3 6 12)')
    
    parser.add_argument('--datasets', nargs='+',
                        choices=['regular', 'interp', 'dynamic', 'brent3covid'],
                        default=['regular', 'interp'],
                        help='Datasets to use (default: regular interp)')
    
    parser.add_argument('--timesteps', nargs='+', type=int,
                        default=[6, 12, 18],
                        help='Timesteps for RNN models (default: 6 12 18)')
    
    # CSV output control
    csv_group = parser.add_mutually_exclusive_group()
    csv_group.add_argument('--save-csv', action='store_true', default=True,
                          help='Save results to CSV files (default)')
    csv_group.add_argument('--no-csv', action='store_true', default=False,
                          help='Do not save results to CSV files')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 Starting Comprehensive Model Training")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Prediction Lags: {args.prediction_lags}")
    print(f"Datasets: {args.datasets}")
    print(f"Timesteps (for RNNs): {args.timesteps}")
    print(f"Save CSV: {not args.no_csv}")
    print("="*60)
    
    trainer = ComprehensiveModelTrainer()
    
    # Convert command line args to method parameters
    trainer.train_all(
        models=args.models,
        dataset_types=tuple(args.datasets), 
        lags=tuple(args.prediction_lags), 
        timesteps=tuple(args.timesteps)
    )
    
    # Save results with CSV control
    trainer.save_results(save_csv=not args.no_csv)
    
    print(f"\n🎉 Comprehensive training complete!")
    print(f"   Total models trained: {len(trainer.results)}")
    if not args.no_csv:
        print("   Results saved to CSV files")


if __name__ == "__main__":
    main()

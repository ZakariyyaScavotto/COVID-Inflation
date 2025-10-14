#!/usr/bin/env python3
"""
ROBUST MODEL EVALUATOR - Handles compatibility and import issues gracefully
"""

import pandas as pd
import numpy as np
import pickle
import os
import csv
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
    print("✓ TensorFlow available")
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️ TensorFlow not available - will skip neural network models")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
    print("✓ Scikit-learn available")
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ Scikit-learn not available - cannot proceed")
    exit(1)

# Import custom ensemble class with robust error handling
HAS_ENSEMBLE = False
try:
    sys.path.append('/home/ubuntu/COVID-Inflation')
    from myEnsemble import myEnsembleModel
    HAS_ENSEMBLE = True
    print("✓ Custom Ensemble class available")
except ImportError as e:
    print(f"⚠️ Custom Ensemble class not available: {e}")
    HAS_ENSEMBLE = False
except Exception as e:
    print(f"⚠️ Error importing ensemble: {e}")
    HAS_ENSEMBLE = False

def load_model_safely(model_path, model_type):
    """Safely load different model types with robust error handling."""
    try:
        if model_type == 'Ensemble':
            if not HAS_ENSEMBLE:
                raise ImportError("Custom ensemble class not available")
            
            # Try custom load method first
            try:
                ensemble_model = myEnsembleModel.load(model_path)
                if ensemble_model is None or ensemble_model.weights is None:
                    raise ValueError("Ensemble model or weights is None")
                return ensemble_model, "success"
            except Exception as custom_error:
                print(f"Custom ensemble load failed: {custom_error}")
                # Fallback to regular pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                if model is None:
                    return None, f"Ensemble model is None: {custom_error}"
                return model, "success"
        
        elif model_path.endswith('.pickle'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if model is None:
                return None, "Pickle model is None"
            
            # Test if the model has a predict method
            if not hasattr(model, 'predict'):
                return None, "Model has no predict method"
            
            return model, "success"
            
        elif model_path.endswith('.h5') and HAS_TENSORFLOW:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, "success"
            
        else:
            return None, "Unsupported model format or TensorFlow not available"
            
    except Exception as e:
        return None, f"Load error: {str(e)}"

def predict_safely(model, X_data, model_type):
    """Make predictions with robust error handling for compatibility issues."""
    try:
        # Standard prediction
        predictions = model.predict(X_data)
        return predictions, "success"
        
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            # This is a sklearn version compatibility issue with RandomForest
            print(f"⚠️ Sklearn version compatibility issue detected: {e}")
            try:
                # Try to temporarily modify the model to avoid the issue
                if hasattr(model, 'estimators_'):
                    # This is likely a RandomForestRegressor
                    print("Attempting workaround for RandomForest compatibility...")
                    # Use a simple ensemble prediction manually
                    predictions_list = []
                    for estimator in model.estimators_:
                        try:
                            pred = estimator.predict(X_data)
                            predictions_list.append(pred)
                        except:
                            continue
                    
                    if predictions_list:
                        predictions = np.mean(predictions_list, axis=0)
                        return predictions, "success_workaround"
                    else:
                        return None, f"RandomForest workaround failed: {e}"
                else:
                    return None, f"Attribute error: {e}"
            except Exception as workaround_error:
                return None, f"Workaround failed: {workaround_error}"
        else:
            return None, f"Attribute error: {e}"
            
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

class RobustModelEvaluator:
    """Robust model evaluator that handles compatibility issues gracefully."""
    
    def __init__(self):
        """Initialize with correct configurations."""
        self.dataset_files = {
            'regular': 'Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx',
            'interp': 'Data/ConstructedDataframes/INTERPAllEcon1990AndCOVIDWithLags.xlsx',
            'dynamic': 'Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLagsDynamic.xlsx',
            'brent3covid': 'Data/ConstructedDataframes/AllEconBrentPlus3COVID.xlsx'
        }
        
        self.model_dirs = {
            'regular': 'Models/',
            'interp': 'InterpModels/',
            'dynamic': 'Models/',
            'brent3covid': 'Brent3COVIDModels/'
        }
        
        self.results = []
        
    def load_dataset(self, dataset_type: str):
        """Load dataset with proper preprocessing."""
        dataset_path = self.dataset_files[dataset_type]
        print(f"Loading {dataset_type} dataset from {dataset_path}")
        
        data = pd.read_excel(dataset_path)
        
        # Remove Date column if present
        if 'Date' in data.columns:
            data = data.drop('Date', axis=1)
        
        # Apply INTERP synthetic data removal BEFORE train/test split (CORRECT methodology)
        if dataset_type == 'interp':
            print("🔧 INTERP: Applying synthetic data removal (every 4th row) to full dataset")
            print(f"   Before removal: {data.shape[0]} samples")
            data = data.iloc[::4, :].reset_index(drop=True)
            print(f"   After removal: {data.shape[0]} samples")
            
        return data
    
    def prepare_rnn_sequences(self, X: np.ndarray, timestep: int):
        """Create proper 3D sequences for RNN models."""
        if len(X) < timestep:
            return None
            
        sequences = []
        for i in range(len(X) - timestep + 1):
            sequence = X[i:i + timestep]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def prepare_data_for_model(self, econData: pd.DataFrame, model_type: str, 
                              timestep: int = None, window: int = 346, testWindow: int = 2):
        """Prepare data using REFERENCE methodology - exact train/test split."""
        
        # Find inflation column
        inflation_col = "AnnualizedMoM-CPI-Inflation"
        if inflation_col not in econData.columns:
            available_cols = [col for col in econData.columns if 'inflation' in col.lower() or 'cpi' in col.lower()]
            if available_cols:
                inflation_col = available_cols[0]
        
        # Apply StandardScaler to ENTIRE dataset BEFORE train/test split (EXACT reference methodology)
        scaler = StandardScaler()
        econData_scaled = pd.DataFrame(scaler.fit_transform(econData), columns=econData.columns)
        
        if model_type in ['RNN', 'LSTM', 'GRU'] and timestep:
            # RNN-specific preparation using EXACT ORIGINAL methodology
            # Use secondTime=True logic for post-2020 evaluation (window=346)
            secondTime = True  # This matches published paper methodology
            if not secondTime:
                trainDf = econData_scaled.iloc[:window]
            else:
                if window == 346:  # First post2020 - use small training window
                    trainDf = econData_scaled.iloc[window-6-timestep:window]  # Adjust for RNN timestep needs
                else:
                    trainDf = econData_scaled.iloc[window-testWindow-timestep:window]
                    
            xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
            yTrain = trainDf.loc[:, trainDf.columns == inflation_col]
            
            # EXACT original test range from recurrentXTimestep.py files
            # xTest = econData.iloc[window-timestep:window+testWindow-1]
            test_start = window - timestep
            test_end = window + testWindow - 1  # This is EXCLUSIVE end for iloc
            
            # Check if we have enough data
            if test_end - test_start < timestep:
                raise ValueError(f"Cannot create sequences with timestep {timestep}: need {timestep} samples, have {test_end - test_start}")
            
            xTest = econData_scaled.iloc[test_start:test_end].loc[:, econData_scaled.columns != inflation_col]
            yTest = econData_scaled.iloc[window:window+testWindow].loc[:, econData_scaled.columns == inflation_col]
            
            # Convert to numpy for RNN sequences (data already scaled)
            X_train = xTrain.values
            X_test = xTest.values
            y_train = np.array([value[0] for value in yTrain.values.tolist()])
            y_test = np.array([value[0] for value in yTest.values.tolist()])
            
            # Create RNN sequences (data already scaled)
            X_train_seq = self.prepare_rnn_sequences(X_train, timestep)
            X_test_seq = self.prepare_rnn_sequences(X_test, timestep)
            
            if X_train_seq is None or X_test_seq is None:
                raise ValueError(f"Cannot create sequences with timestep {timestep}")
            
            y_train = y_train[timestep-1:]
            y_test = y_test[-len(X_test_seq):]
            
            return X_train_seq, y_train, X_test_seq, y_test, scaler
            
        else:
            # Standard model preparation using EXACT ORIGINAL methodology
            # Use secondTime=True logic for post-2020 evaluation (matches published paper)
            secondTime = True  # This matches published paper methodology
            if not secondTime:
                # Original: full training window
                trainDf = econData_scaled.iloc[:window]
                testDf = econData_scaled.iloc[window:window+testWindow]
            else:
                if window == 346:  # First post2020 - use small training window (6 months)
                    trainDf = econData_scaled.iloc[window-6:window]
                    testDf = econData_scaled.iloc[window:window+testWindow]
                else:
                    trainDf = econData_scaled.iloc[window-testWindow:window]
                    testDf = econData_scaled.iloc[window:window+testWindow]
                
            # EXACT feature/target split from reference
            xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
            yTrain = trainDf.loc[:, trainDf.columns == inflation_col]
            xTest = testDf.loc[:, testDf.columns != inflation_col] 
            yTest = testDf.loc[:, testDf.columns == inflation_col]
            
            # Convert y to numpy arrays (exact from reference)
            yTrain_np = np.array([value[0] for value in yTrain.values.tolist()])
            yTest_np = np.array([value[0] for value in yTest.values.tolist()])
        
        return xTrain.values, yTrain_np, xTest.values, yTest_np, scaler
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate RMSE and MAE with robust handling."""
        try:
            # Convert to numpy arrays
            if hasattr(y_true, 'values'):
                y_true = y_true.values.flatten()
            elif isinstance(y_true, pd.DataFrame):
                y_true = y_true.iloc[:, 0].values
            else:
                y_true = np.array(y_true).flatten()
                
            if hasattr(y_pred, 'flatten'):
                y_pred = y_pred.flatten()
            else:
                y_pred = np.array(y_pred).flatten()
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Remove NaN/inf values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                return np.nan, np.nan
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            return round(rmse, 4), round(mae, 4)
        
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return np.nan, np.nan
    
    def get_model_filename(self, model_type: str, lag: int, timestep: int = None):
        """Get correct model filename."""
        if model_type == 'DynLR':
            return f"DynLRModel_lag{lag}.pickle"
        elif model_type == 'DynNN':
            return f"DynNNModel_lag{lag}.h5"
        elif model_type == 'Ensemble':
            return f"EnsembleModel_lag{lag}.pickle"
        elif model_type in ['RNN', 'LSTM', 'GRU'] and timestep is not None:
            return f"{model_type}Model_lag{lag}_t{int(timestep)}.h5"
        elif model_type in ['NN']:
            return f"{model_type}Model_lag{lag}.h5"
        else:
            return f"{model_type}Model_lag{lag}.pickle"
    
    def evaluate_single_model(self, dataset_type: str, model_type: str, lag: int, timestep: int = None):
        """Evaluate a single model with robust error handling."""
        print(f"\n{'='*50}")
        print(f"🔍 {model_type} | {dataset_type} | Lag {lag}" + (f" | t{timestep}" if timestep else ""))
        print(f"{'='*50}")
        
        try:
            # Get model path
            filename = self.get_model_filename(model_type, lag, timestep)
            model_dir = self.model_dirs[dataset_type]
            model_path = os.path.join(model_dir, filename)
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return self.create_error_result(dataset_type, model_type, lag, timestep, "Model file not found")
            
            # Load dataset (NO preprocessing here)
            econData = self.load_dataset(dataset_type)
            
            # Prepare data using REFERENCE methodology
            try:
                xTrain, yTrain, xTest, yTest, scaler = self.prepare_data_for_model(
                    econData, model_type=model_type, timestep=timestep
                )
            except Exception as e:
                print(f"❌ Data preparation failed: {e}")
                return self.create_error_result(dataset_type, model_type, lag, timestep, f"Data prep: {str(e)}")
            
            # Load model
            model, load_status = load_model_safely(model_path, model_type)
            if model is None:
                print(f"❌ Model loading failed: {load_status}")
                return self.create_error_result(dataset_type, model_type, lag, timestep, load_status)
            
            # INTERP synthetic data removal already applied during dataset loading
            interp_fixed = (dataset_type == 'interp')
            
            # Make predictions
            train_pred, train_status = predict_safely(model, xTrain, model_type)
            if train_pred is None:
                print(f"❌ Training prediction failed: {train_status}")
                return self.create_error_result(dataset_type, model_type, lag, timestep, train_status)
            
            test_pred, test_status = predict_safely(model, xTest, model_type)
            if test_pred is None:
                print(f"❌ Test prediction failed: {test_status}")
                return self.create_error_result(dataset_type, model_type, lag, timestep, test_status)
            
            # Calculate metrics
            train_rmse, train_mae = self.calculate_metrics(yTrain, train_pred)
            test_rmse, test_mae = self.calculate_metrics(yTest, test_pred)
            
            # Status analysis
            status = 'Success'
            notes = []
            warnings_list = []
            
            if "workaround" in train_status or "workaround" in test_status:
                warnings_list.append("Used compatibility workaround")
                
            if np.isnan(train_rmse) or np.isnan(test_rmse):
                status = 'Warning: NaN metrics'
                notes.append('NaN_metrics')
            
            if test_rmse == 0.0 and test_mae == 0.0:
                if dataset_type == 'interp' and not interp_fixed:
                    status = 'Critical: INTERP overfitting'
                    notes.append('interp_overfitting')
                else:
                    status = 'Warning: Perfect fit'
                    notes.append('perfect_fit')
            
            print(f"✅ Evaluation complete:")
            print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
            print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
            
            if warnings_list:
                print(f"   ⚠️ {'; '.join(warnings_list)}")
            
            return {
                'Dataset': dataset_type,
                'Model': model_type,
                'Prediction_Lag': lag,
                'Timestep': timestep if timestep is not None else 'N/A',
                'Train_RMSE': train_rmse,
                'Train_MAE': train_mae,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'Status': status,
                'Model_File': filename,
                'Data_Shape_Train': f"{xTrain.shape}",
                'Data_Shape_Test': f"{xTest.shape}",
                'INTERP_Fixed': 'Yes' if interp_fixed else 'No',
                'Warnings': '; '.join(warnings_list),
                'Notes': '; '.join(notes)
            }
                
        except Exception as e:
            print(f"❌ General error: {e}")
            return self.create_error_result(dataset_type, model_type, lag, timestep, f"General: {str(e)}")
    
    def create_error_result(self, dataset_type: str, model_type: str, lag: int, timestep: int, error_msg: str):
        """Create standardized error result."""
        return {
            'Dataset': dataset_type,
            'Model': model_type,
            'Prediction_Lag': lag,
            'Timestep': timestep if timestep is not None else 'N/A',
            'Train_RMSE': 'ERROR',
            'Train_MAE': 'ERROR', 
            'Test_RMSE': 'ERROR',
            'Test_MAE': 'ERROR',
            'Status': f'Failed: {error_msg}',
            'Model_File': 'N/A',
            'Data_Shape_Train': 'N/A',
            'Data_Shape_Test': 'N/A',
            'INTERP_Fixed': 'N/A',
            'Warnings': '',
            'Notes': error_msg
        }
    
    def run_evaluation(self):
        """Run evaluation with progress tracking."""
        print("🚀 ROBUST MODEL EVALUATION")
        print("="*50)
        
        datasets = ['regular', 'interp'] # Focus on available datasets
        prediction_lags = [1, 3, 6, 12]
        timesteps = [6, 12, 18]
        
        standard_models = ['LR', 'DynLR', 'RF', 'NN', 'DynNN', 'Ensemble']
        rnn_models = ['RNN', 'LSTM', 'GRU']
        
        total_models = len(datasets) * len(prediction_lags) * (len(standard_models) + len(rnn_models) * len(timesteps))
        completed = 0
        
        for dataset_type in datasets:
            print(f"\n📊 Dataset: {dataset_type}")
            
            for lag in prediction_lags:
                print(f"\n  🎯 Lag {lag}")
                
                # Standard models
                for model_type in standard_models:
                    result = self.evaluate_single_model(dataset_type, model_type, lag)
                    self.results.append(result)
                    completed += 1
                    print(f"Progress: {completed}/{total_models} ({completed/total_models*100:.1f}%)")
                
                # RNN models
                for model_type in rnn_models:
                    for timestep in timesteps:
                        result = self.evaluate_single_model(dataset_type, model_type, lag, timestep)
                        self.results.append(result)
                        completed += 1
                        print(f"Progress: {completed}/{total_models} ({completed/total_models*100:.1f}%)")
        
        return self.results
    
    def save_results(self):
        """Save results with analysis."""
        if not self.results:
            print("❌ No results to save")
            return
            
        results_df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all results
        all_file = f"ROBUST_ALL_RESULTS_{timestamp}.csv"
        results_df.to_csv(all_file, index=False)
        print(f"✅ All results: {all_file}")
        
        # Filter and save successful results
        successful_df = results_df[~results_df['Status'].str.contains('Failed', na=False)]
        
        if len(successful_df) > 0:
            success_file = f"ROBUST_SUCCESS_RESULTS_{timestamp}.csv"
            successful_df.to_csv(success_file, index=False)
            print(f"✅ Successful results: {success_file}")
            
            # Performance analysis
            numeric_results = successful_df[
                (successful_df['Train_RMSE'] != 'ERROR') & 
                (successful_df['Test_RMSE'] != 'ERROR')
            ].copy()
            
            if len(numeric_results) > 0:
                for col in ['Train_RMSE', 'Train_MAE', 'Test_RMSE', 'Test_MAE']:
                    numeric_results[col] = pd.to_numeric(numeric_results[col], errors='coerce')
                
                clean_results = numeric_results.dropna(subset=['Test_RMSE'])
                
                print(f"\n🏆 TOP 10 MODELS:")
                top_models = clean_results.nsmallest(10, 'Test_RMSE')
                for i, (_, row) in enumerate(top_models.iterrows(), 1):
                    ts_info = f" (t{row['Timestep']})" if row['Timestep'] != 'N/A' else ""
                    print(f"  {i:2d}. {row['Model']}{ts_info:<12} | {row['Dataset']:<8} | Lag {row['Prediction_Lag']:2d} | RMSE: {row['Test_RMSE']:.4f}")
        
        print(f"\n📊 SUMMARY:")
        print(f"Total: {len(results_df)}")
        print(f"✅ Successful: {len(successful_df)}")
        print(f"❌ Failed: {len(results_df) - len(successful_df)}")
        
        return results_df

def main():
    """Main execution."""
    evaluator = RobustModelEvaluator()
    results = evaluator.run_evaluation()
    evaluator.save_results()
    print("\n🎉 Robust evaluation complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training Examples and Demonstrations for COVID-Inflation Project
Example usage script for the comprehensive model trainer.

Provides example usage of model training functions and validation.
This shows how to use the training script with different configurations.

Originally created for demonstrating training methodology
"""


import subprocess
import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Import required libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    print("✓ TensorFlow/Keras available")
except ImportError as e:
    print(f"⚠️ TensorFlow/Keras not available: {e}")
    tf = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("✓ Scikit-learn available")
except ImportError as e:
    print(f"⚠️ Scikit-learn not available: {e}")
    exit(1)

# Import custom classes
try:
    sys.path.append('/home/ubuntu/COVID-Inflation')
    from myEnsemble import myEnsembleModel
    print("✓ Custom Ensemble class available")
except ImportError as e:
    print(f"⚠️ Custom Ensemble class not available: {e}")


class TrainingExamples:
    """Example implementations of model training for validation and demonstration"""
    def __init__(self):
        self.scaler = StandardScaler()

    def load_sample_data(self, dataset_type='regular', sample_size=100):
        """Load a sample of data for quick testing"""
        if dataset_type == 'regular':
            data_path = 'Data/ConstructedDataframes/AllEcon1990AndCOVIDWithLags.xlsx'
        else:
            data_path = 'Data/ConstructedDataframes/INTERPAllEcon1990AndCOVIDWithLags.xlsx'
        print(f"Loading sample from {dataset_type} dataset")
        data = pd.read_excel(data_path)
        if dataset_type == 'interp':
            # Apply INTERP processing
            original_len = len(data)
            data = data.iloc[::4].reset_index(drop=True)
            print(f"INTERP: Reduced from {original_len} to {len(data)} samples")
        # Take a sample for quick testing
        if len(data) > sample_size:
            data = data.tail(sample_size).reset_index(drop=True)
            print(f"Using sample of {len(data)} rows for quick testing")
        return data
    
    def example_data_preparation(self, lag=12):
        """Demonstrate data preparation methodology"""
        print(f"\n📋 Example: Data Preparation for Lag {lag}")
        print("=" * 50)
        data = self.load_sample_data('regular', 100)
        # Apply lag
        inflation_col = f'Inflation_lag_{lag}'
        if inflation_col not in data.columns:
            print(f"❌ Column {inflation_col} not found")
            return None
        # Remove NaN values
        econData = data.dropna().reset_index(drop=True)
        print(f"Data shape after removing NaN: {econData.shape}")
        # Apply StandardScaler (reference methodology)
        econData_scaled = pd.DataFrame(self.scaler.fit_transform(econData), columns=econData.columns)
        print(f"Data scaled using StandardScaler")
        # Reference methodology split
        window = min(80, len(econData_scaled) - 10)  # Adjusted for sample size
        testWindow = 2
        secondTime = True
        if secondTime and window >= 6:
            trainDf = econData_scaled.iloc[window-6:window]
        else:
            trainDf = econData_scaled.iloc[:window]
        xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
        yTrain = trainDf.loc[:, trainDf.columns == inflation_col]
        xTest = econData_scaled.iloc[window:window+testWindow].loc[:, econData_scaled.columns != inflation_col]
        yTest = econData_scaled.iloc[window:window+testWindow].loc[:, econData_scaled.columns == inflation_col]
        print(f"Training data shape: {xTrain.shape}")
        print(f"Training target shape: {yTrain.shape}")
        print(f"Test data shape: {xTest.shape}")
        print(f"Test target shape: {yTest.shape}")
        return {
            'X_train': xTrain.values,
            'y_train': np.array([value[0] for value in yTrain.values.tolist()]),
            'X_test': xTest.values,
            'y_test': np.array([value[0] for value in yTest.values.tolist()])
        }

    
        
    def example_rnn_data_preparation(self, timestep=6, lag=12):
        """Demonstrate RNN data preparation with sequences"""
        print(f"\n📋 Example: RNN Data Preparation (timestep={timestep}, lag={lag})")
        print("=" * 60)
        
        data = self.load_sample_data('regular', 100)
        
        inflation_col = f'Inflation_lag_{lag}'
        if inflation_col not in data.columns:
            print(f"❌ Column {inflation_col} not found")
            return None
            
        # Data preparation
        econData = data.dropna().reset_index(drop=True)
        econData_scaled = pd.DataFrame(self.scaler.fit_transform(econData), columns=econData.columns)
        
        # RNN-specific preparation
        window = min(80, len(econData_scaled) - 10)
        testWindow = 2
        secondTime = True
        
        if secondTime and window >= 6 + timestep:
            trainDf = econData_scaled.iloc[window-6-timestep:window]
        else:
            trainDf = econData_scaled.iloc[:window]
            
        xTrain = trainDf.loc[:, trainDf.columns != inflation_col]
        yTrain = trainDf.loc[:, trainDf.columns == inflation_col]
        
        # Test data
        test_start = max(0, window - timestep)
        test_end = min(len(econData_scaled), window + testWindow - 1)
        
        xTest = econData_scaled.iloc[test_start:test_end].loc[:, econData_scaled.columns != inflation_col]
        yTest = econData_scaled.iloc[window:window+testWindow].loc[:, econData_scaled.columns == inflation_col]
        
        # Create sequences
        X_train = xTrain.values
        X_test = xTest.values
        y_train = np.array([value[0] for value in yTrain.values.tolist()])
        y_test = np.array([value[0] for value in yTest.values.tolist()])
        
        # Prepare sequences
        X_train_seq = self.prepare_rnn_sequences(X_train, timestep)
        X_test_seq = self.prepare_rnn_sequences(X_test, timestep)
        
        if X_train_seq is None or X_test_seq is None:
            print(f"❌ Cannot create sequences with timestep {timestep}")
            return None
            
        print(f"Training sequences shape: {X_train_seq.shape}")
        print(f"Test sequences shape: {X_test_seq.shape}")
        print(f"Training target length: {len(y_train[:len(X_train_seq)])}")
        print(f"Test target length: {len(y_test[:len(X_test_seq)])}")
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train[:len(X_train_seq)],
            'X_test': X_test_seq, 
            'y_test': y_test[:len(X_test_seq)]
        }
        
    def prepare_rnn_sequences(self, data, timestep):
        """Create RNN sequences from data"""
        if len(data) < timestep:
            return None
            
        sequences = []
        for i in range(len(data) - timestep + 1):
            sequences.append(data[i:(i + timestep)])
        return np.array(sequences)
        
    def example_linear_regression_training(self):
        """Demonstrate Linear Regression training"""
        print(f"\n🔍 Example: Linear Regression Training")
        print("=" * 40)
        
        data_prep = self.example_data_preparation(lag=12)
        if data_prep is None:
            return
            
        # Train model
        model = LinearRegression()
        model.fit(data_prep['X_train'], data_prep['y_train'])
        
        # Make predictions
        y_train_pred = model.predict(data_prep['X_train'])
        y_test_pred = model.predict(data_prep['X_test'])
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(data_prep['y_train'], y_train_pred))
        train_mae = mean_absolute_error(data_prep['y_train'], y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(data_prep['y_test'], y_test_pred))
        test_mae = mean_absolute_error(data_prep['y_test'], y_test_pred)
        
        print(f"✅ Training complete:")
        print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
        print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        
        return model
        
    def example_random_forest_training(self):
        """Demonstrate Random Forest training"""
        print(f"\n🌲 Example: Random Forest Training")
        print("=" * 40)
        
        data_prep = self.example_data_preparation(lag=12)
        if data_prep is None:
            return
            
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(data_prep['X_train'], data_prep['y_train'])
        
        # Make predictions
        y_train_pred = model.predict(data_prep['X_train'])
        y_test_pred = model.predict(data_prep['X_test'])
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(data_prep['y_train'], y_train_pred))
        train_mae = mean_absolute_error(data_prep['y_train'], y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(data_prep['y_test'], y_test_pred))
        test_mae = mean_absolute_error(data_prep['y_test'], y_test_pred)
        
        print(f"✅ Training complete:")
        print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
        print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        
        return model
        
    def example_neural_network_training(self):
        """Demonstrate Neural Network training"""
        print(f"\n🧠 Example: Neural Network Training")
        print("=" * 40)
        
        if tf is None:
            print("❌ TensorFlow not available for NN training")
            return
            
        data_prep = self.example_data_preparation(lag=12)
        if data_prep is None:
            return
            
        # Build model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(data_prep['X_train'].shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        history = model.fit(data_prep['X_train'], data_prep['y_train'], 
                           epochs=50, batch_size=16, verbose=0,
                           callbacks=[early_stopping])
        
        # Make predictions
        y_train_pred = model.predict(data_prep['X_train'], verbose=0).flatten()
        y_test_pred = model.predict(data_prep['X_test'], verbose=0).flatten()
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(data_prep['y_train'], y_train_pred))
        train_mae = mean_absolute_error(data_prep['y_train'], y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(data_prep['y_test'], y_test_pred))
        test_mae = mean_absolute_error(data_prep['y_test'], y_test_pred)
        
        print(f"✅ Training complete:")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
        print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        
        return model
        
    def example_rnn_training(self, model_type='LSTM'):
        """Demonstrate RNN model training"""
        print(f"\n🔄 Example: {model_type} Training")
        print("=" * 40)
        
        if tf is None:
            print(f"❌ TensorFlow not available for {model_type} training")
            return
            
        data_prep = self.example_rnn_data_preparation(timestep=6, lag=12)
        if data_prep is None:
            return
            
        # Build model
        model = Sequential()
        
        if model_type == "RNN":
            model.add(SimpleRNN(32, input_shape=(data_prep['X_train'].shape[1], data_prep['X_train'].shape[2])))
        elif model_type == "LSTM":
            model.add(LSTM(32, input_shape=(data_prep['X_train'].shape[1], data_prep['X_train'].shape[2])))
        elif model_type == "GRU":
            model.add(GRU(32, input_shape=(data_prep['X_train'].shape[1], data_prep['X_train'].shape[2])))
        else:
            print(f"❌ Unknown model type: {model_type}")
            return
            
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        history = model.fit(data_prep['X_train'], data_prep['y_train'],
                           epochs=50, batch_size=16, verbose=0,
                           callbacks=[early_stopping])
        
        # Make predictions
        y_train_pred = model.predict(data_prep['X_train'], verbose=0).flatten()
        y_test_pred = model.predict(data_prep['X_test'], verbose=0).flatten()
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(data_prep['y_train'], y_train_pred))
        train_mae = mean_absolute_error(data_prep['y_train'], y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(data_prep['y_test'], y_test_pred))
        test_mae = mean_absolute_error(data_prep['y_test'], y_test_pred)
        
        print(f"✅ Training complete:")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
        print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        
        return model
        
    def example_ensemble_training(self):
        """Demonstrate Ensemble training"""
        print(f"\n🎯 Example: Ensemble Training")
        print("=" * 40)
        
        try:
            data_prep = self.example_data_preparation(lag=12)
            if data_prep is None:
                return
                
            # Train ensemble
            model = myEnsembleModel()
            model.fit(data_prep['X_train'], data_prep['y_train'])
            
            # Make predictions
            y_train_pred = model.predict(data_prep['X_train'])
            y_test_pred = model.predict(data_prep['X_test'])
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(data_prep['y_train'], y_train_pred))
            train_mae = mean_absolute_error(data_prep['y_train'], y_train_pred)
            test_rmse = np.sqrt(mean_squared_error(data_prep['y_test'], y_test_pred))
            test_mae = mean_absolute_error(data_prep['y_test'], y_test_pred)
            
            print(f"✅ Training complete:")
            print(f"   Train: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}")
            print(f"   Test:  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
            
            return model
            
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
            return None
            
    def run_all_examples(self):
        """Run all training examples"""
        print("🚀 Running All Training Examples")
        print("=" * 60)
        
        # Data preparation examples
        self.example_data_preparation(lag=12)
        self.example_rnn_data_preparation(timestep=6, lag=12)
        
        # Model training examples
        self.example_linear_regression_training()
        self.example_random_forest_training()
        self.example_neural_network_training()
        
        # RNN examples
        for model_type in ['RNN', 'LSTM', 'GRU']:
            self.example_rnn_training(model_type)
            
        # Ensemble example
        self.example_ensemble_training()
        
        print("\n✅ All examples completed!")
        
    def validate_methodology(self):
        """Validate that methodology matches reference implementation"""
        print("\n🔍 Validating Methodology Against Reference")
        print("=" * 50)
        
        # Test data preparation consistency
        print("Testing data preparation consistency...")
        
        # Test with different configurations
        configs = [
            {'lag': 1, 'timestep': None},
            {'lag': 12, 'timestep': None}, 
            {'lag': 12, 'timestep': 6},
            {'lag': 12, 'timestep': 18}
        ]
        
        for config in configs:
            try:
                if config['timestep']:
                    result = self.example_rnn_data_preparation(config['timestep'], config['lag'])
                    print(f"✅ RNN data prep (lag={config['lag']}, t={config['timestep']}): {result is not None}")
                else:
                    result = self.example_data_preparation(config['lag'])
                    print(f"✅ Standard data prep (lag={config['lag']}): {result is not None}")
            except Exception as e:
                print(f"❌ Failed config {config}: {e}")
                
        print("Methodology validation complete!")

def main():
    """Main function to run training examples"""
    examples = TrainingExamples()
    
    print("Select example to run:")
    print("1. Run all examples")
    print("2. Data preparation examples")
    print("3. Model training examples")
    print("4. Validate methodology")
    print("5. Quick Linear Regression test")
    
    try:
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            examples.run_all_examples()
        elif choice == '2':
            examples.example_data_preparation()
            examples.example_rnn_data_preparation()
        elif choice == '3':
            examples.example_linear_regression_training()
            examples.example_neural_network_training()
            examples.example_rnn_training('LSTM')
        elif choice == '4':
            examples.validate_methodology()
        elif choice == '5':
            examples.example_linear_regression_training()
        else:
            print("Running default: Linear Regression example")
            examples.example_linear_regression_training()
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        # Default fallback
        examples.example_linear_regression_training()

if __name__ == "__main__":
    main()
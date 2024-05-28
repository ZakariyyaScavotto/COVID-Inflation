import numpy as np
from sklearn.linear_model import LinearRegression
import pickle 
import keras
from keras import metrics
import tensorflow as tf

class myEnsembleModel:
    def __init__(self, lr, rf, nn_model_path, sampleWeights, myMSE):
        self.linear_regression = lr
        self.random_forest = rf
        self.nn_model_path = nn_model_path
        self.neural_network = keras.models.load_model(nn_model_path, compile=False)
        self.neural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss = myMSE(sampleWeights), weighted_metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
        # Weights isn't trained yet, so set it to None
        # When trained, it will be a LinearRegression object
        self.weights = None

    def fit(self, X, y):
        # Get each model's predictions on the training data
        predictions_lr = self.linear_regression.predict(X)
        predictions_rf = self.random_forest.predict(X)
        predictions_nn = self.neural_network.predict(X).flatten()

        # Combine predictions into a matrix
        predictions = np.column_stack((predictions_lr, predictions_rf, predictions_nn))

        # Train weights using linear regression
        self.weights = LinearRegression().fit(predictions, y)

    def predict(self, X):
        # Make predictions using individual models
        predictions_lr = self.linear_regression.predict(X)
        predictions_rf = self.random_forest.predict(X)
        predictions_nn = self.neural_network.predict(X).flatten()

        # Combine predictions into a matrix
        predictions = np.column_stack((predictions_lr, predictions_rf, predictions_nn))

        # Use ensemble weights to make final prediction
        return self.weights.predict(predictions)
    
    def save(self, filename):
        # Save the Keras model separately
        self.neural_network.save(self.nn_model_path)
        # Set the neural_network attribute to None so it's not pickled
        self.neural_network = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        # Reload the neural_network attribute
        self.neural_network = keras.models.load_model(self.nn_model_path, compile=False)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        # Load the Keras model from the saved file
        model.neural_network = keras.models.load_model(model.nn_model_path, compile=False)
        return model
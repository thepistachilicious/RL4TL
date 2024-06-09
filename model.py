
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow warnings
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel_FC:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """

        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_fc_model')
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self._learning_rate))
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True,
                   show_layer_names=True)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TrainModel_CNN:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim  # Input dimension (assuming input_dim is a flat dimension, e.g., 80)
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

    def _build_model(self, num_layers, width):
        """
        Build and compile a convolutional neural network
        """
        input_height = 20 # Example height
        input_width = 4 # Example width
        input_channels = 1  # Example channels (grayscale)

        inputs = keras.Input(shape=(input_height, input_width, input_channels))

        # First Convolutional Layer
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1,1))(inputs)
        x = layers.MaxPooling2D((2, 1))(x)

        # Second Convolutional Layer
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
        x = layers.MaxPooling2D((2, 1))(x)

        # Third Convolutional Layer
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1,1))(x)
        x = layers.MaxPooling2D((2, 1))(x)

        # Flatten before passing to Dense layers
        x = layers.Flatten()(x)

        # Fully connected layer
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(32, activation='linear')(x)

        # Output layer
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_cnn_model')
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self._learning_rate))

        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        input_height = 20  # Example height
        input_width = 4 # Example width
        input_channels = 1  # Example channels (grayscale)

        state = np.reshape(state, [1, input_height, input_width, input_channels])  # Adjust shape for CNN
        return self._model.predict(state)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        input_height = 20 # Example height
        input_width = 4  # Example width
        input_channels = 1  # Example channels (grayscale)

        states = np.reshape(states, [-1, input_height, input_width, input_channels])  # Adjust shape for CNN
        return self._model.predict(states)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        input_height = 20 # Example height
        input_width = 4 # Example width
        input_channels = 1  # Example channels (grayscale)

        states = np.reshape(states, [-1, input_height, input_width, input_channels])  # Adjust shape for CNN
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True,
                   show_layer_names=True)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._is_cnn = self._check_if_cnn()

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists.
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def _check_if_cnn(self):
        """
        Check if the model is a CNN by inspecting the input layer.
        """
        for layer in self._model.layers:
            if isinstance(layer, Conv2D):
                return True
        return False

    def predict_one(self, state):
        """
        Predict the action values from a single state.
        Adjust shape based on whether the model is NN or CNN.
        """
        if self._is_cnn:
            # For CNN
            input_height = 20  # Example height
            input_width = 4  # Example width
            input_channels = 1  # Example channels (grayscale)

            state = np.reshape(state, [1, input_height, input_width, input_channels])
        else:
            # For NN
            state = np.reshape(state, [1, self._input_dim])

        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim
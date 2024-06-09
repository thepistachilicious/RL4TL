import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'D:\\AI\\DoAnCK2\\TLCS\\TLCS\\models\\model_7\\trained_model.h5'  # Adjust the path as needed
model = load_model(model_path)

# Plot the model structure
plot_model_path = 'D:\\AI\\DoAnCK2\\TLCS\\TLCS\\models\\model_7\\model_structure.png'  # Adjust the path as needed
plot_model(model, to_file=plot_model_path, show_shapes=True, show_layer_names=True)

print(f'Model plot saved to {plot_model_path}')

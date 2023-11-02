import os
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

# def _data(samples=300, centers=3, features=3, random_state=42):
#     data, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=random_state)
#     scaler = MinMaxScaler()
#     return scaler.fit_transform(data)

def train_som(data, x_size=10, y_size=10, sigma=1.0, lr=0.5, iterations=1000):
    start_time = time.time()
    
    som = MiniSom(x_size, y_size, data.shape[1], sigma=sigma, learning_rate=lr)
    som.random_weights_init(data)
    som.train(data, iterations)
    
    end_time = time.time()
    print(f"SOM training completed in {end_time - start_time:.2f} seconds.")
    return som

def calculate_response(neuron_weight, data_point):
    # Calculate Euclidean distance between the neuron weight and data point
    
    mask = ~np.isnan(neuron_weight) & ~np.isnan(data_point)
    
    # Calculate the Euclidean distance for non-NaN values
    distance = np.linalg.norm(neuron_weight[mask] - data_point[mask])
    return distance  # Negative distance can be used as a measure of similarity

def plot_activation_map_as_heatmap(som, data_point, img_label):
    # Initialize the activation map
    grid_shape = som.get_weights().shape[:2]
#     print('grid_hape',grid_shape)
    som_neurons_weights = som.get_weights() #.reshape(-1, som.get_weights().shape[2])
#     print('som_neurons_weights', som_neurons_weights)
    activation_map = np.zeros((grid_shape[0], grid_shape[1]))
#     print('data point', data_point)
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Calculate the response of neuron (i, j) to the data point
            neuron_weight = som_neurons_weights[i, j]
            response = calculate_response(neuron_weight, data_point)
#             print('response',response)
            # Assign the response to the activation map
            activation_map[i, j] = response
    
    activation_map = np.log1p(activation_map) 
    # max_index = np.argmax(activation_map)
    # max_row, max_col = np.unravel_index(max_index, activation_map.shape)
    # print("Maximum Value:", activation_map[max_row, max_col])
    # print("Location (Row, Column):", max_row, max_col)

    
    # Create a heatmap of the activation map
    plt.figure(figsize=(8, 8))
    plt.imshow(activation_map, cmap='viridis', interpolation='none', aspect='auto')#,vmin=global_min, vmax=global_max)
    plt.colorbar()  # Add a colorbar to show the scale
    # plt.title('Activation Map (Heatmap) for Data Point')
    # plt.xlabel('SOM Neuron X-coordinate')
    # plt.ylabel('SOM Neuron Y-coordinate')
    # Save the heatmap as an image
    plt.savefig("data/heatmaps/heatmap-dp{}".format(img_label+1), bbox_inches='tight')
    plt.close()  # Close the plot to free up resources


# Usage
data = pd.read_csv('data/descriptor/all_descriptors.csv')
if not isinstance(data, np.ndarray):
    data = data.values

# from sklearn.datasets import make_blobs   
# data, _ = make_blobs(n_samples=2000, centers=5, n_features=400)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# data_scaled = data

som = train_som(data_scaled, x_size=28, y_size=28, sigma=0.5, lr=0.05, iterations=1000)


global_min = 0
global_max = 12
for i in range(data_scaled.shape[0]):
    print(f"Processing data point {i}...",end='\r')
    plot_activation_map_as_heatmap(som, data_scaled[i], img_label = i)
    if i > 30:
        break

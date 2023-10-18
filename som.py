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

def calculate_feature_heatmaps(data, som):
    start_time = time.time()
    feature_heatmaps = np.zeros((som._weights.shape[0], som._weights.shape[1], data.shape[1]))
    counts = np.zeros((som._weights.shape[0], som._weights.shape[1]))

    for i, xx in enumerate(data):
        if i % 20 == 0:
            print(f"Processing data point {i+1}/{data.shape[0]}", end="\r")
        bmu = som.winner(xx)
        feature_heatmaps[bmu[0], bmu[1], :] += xx
        counts[bmu[0], bmu[1]] += 1
        
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            if counts[i, j] > 0:
                feature_heatmaps[i, j, :] /= counts[i, j]
                
    end_time = time.time()
    print(f"Feature heatmaps calculation completed in {end_time - start_time:.2f} seconds.")
    return feature_heatmaps

def save_heatmaps(feature_heatmaps, output_folder):
    start_time = time.time()
    
    num_features = feature_heatmaps.shape[2]
    
    for i in range(num_features):
        plt.figure(figsize=(28, 28))
        im = plt.imshow(feature_heatmaps[:, :, i], cmap="viridis", interpolation='none')
        plt.title(f'Feature {i+1} Heatmap')
        plt.colorbar(im, orientation='vertical', fraction=0.045, pad=0.05).set_label('Feature Average')
        
        # Save each heatmap as an image file
        plt.savefig(os.path.join(output_folder, f'feature_{i+1}_heatmap.png'))
        
        # Close the figure to free up memory
        plt.close()
        
        if i % 2 == 0:
            print(f"Saving heatmap {i+1}/{num_features}", end="\r")
        

    end_time = time.time()
    print(f"Heatmap saving completed in {end_time - start_time:.2f} seconds.")

# Usage
data = pd.read_csv('data/descriptor/all_descriptors_cut.csv')
if not isinstance(data, np.ndarray):
    data = data.values

# from sklearn.datasets import make_blobs   
# data, _ = make_blobs(n_samples=2000, centers=5, n_features=400)

# scaler = MinMaxScaler()
# data_scaled = scaler.fit_transform(data)

data_scaled = data

som = train_som(data_scaled, x_size=28, y_size=28, sigma=0.5, lr=0.1, iterations=2000)
feature_heatmaps = calculate_feature_heatmaps(data_scaled, som)
save_heatmaps(feature_heatmaps, "data/heatmaps/")

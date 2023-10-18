import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def load_dataset(file_path):
    data = pd.read_csv(file_path)

    # Load in feature names from CSV file
    feature_names = pd.read_csv('feature_names.csv')['Value']

    # Set feature names as column names of data dataset
    data.columns = feature_names

    return data

def missing_data_analysis(df):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    return missing_info

def visualize_missing_data(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Visualization')
    plt.show()

def feature_correlation_analysis(df):
    correlation_matrix = df.corr()
    return correlation_matrix

def visualize_feature_correlations(correlation_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.show()

def feature_importance_analysis(X, y):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, y)

    feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
    feature_importance_sorted = feature_importance.sort_values(ascending=False)
    return feature_importance_sorted

def visualize_feature_importance(feature_importance_sorted):
    plt.figure(figsize=(10, 6))
    feature_importance_sorted.plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.show()

# Usage example:
file_path = 'data/descriptor/all_descriptors.csv'
df = load_dataset(file_path)

zero_percentage = (df == 0).mean() * 100
print("Percentage of Zero Values in Each Column:", zero_percentage)

plt.figure(figsize=(8, 6))
plt.hist(zero_percentage, bins=10, edgecolor='k', alpha=0.7)
plt.xlabel('Percentage of Zeros')
plt.ylabel('Frequency')
plt.title('Distribution of Zero Percentages in Columns')
plt.grid(True)
plt.show()
plt.savefig('histogram.png')
plt.close()






# correlation_matrix = feature_correlation_analysis(df)
# visualize_feature_correlations(correlation_matrix)
# X = df.drop('target_column', axis=1)
# y = df['target_column']
# feature_importance_sorted = feature_importance_analysis(X, y)
# visualize_feature_importance(feature_importance_sorted)

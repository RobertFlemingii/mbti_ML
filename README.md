# mbti_ML

Data Analysis and Visualization README
This repository contains a Python script for data analysis and visualization using the matplotlib, numpy, and pandas libraries. The script includes functions for creating distribution graphs, correlation matrices, and scatter plots for a given dataset. The primary focus is on exploring and understanding the distribution and relationships within numerical columns of the data.

Prerequisites
Make sure you have the following Python libraries installed:
pip install matplotlib numpy pandas

Usage
1. Distribution Graphs
The plotPerColumnDistribution function generates distribution graphs (histograms or bar plots) for numerical columns in a DataFrame. It provides a quick overview of the data distribution.

Function Signature:
plotPerColumnDistribution(df, nGraphShown, nGraphPerRow)

Example Usage:
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load your dataset
nRowsRead = 1000  # specify 'None' if you want to read the entire file
df1 = pd.read_csv('your_data.csv', delimiter=',', nrows=nRowsRead)
df1.dataframeName = 'your_data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)

# Example usage:
plotPerColumnDistribution(df1, 10, 5)
2. Correlation Matrix
The plotCorrelationMatrix function generates a correlation matrix heatmap, providing insights into the relationships between numerical columns in the dataset.

Function Signature:
plotCorrelationMatrix(df, graphWidth)
Example Usage:

# Example usage:
plotCorrelationMatrix(df1, 8)

3. Scatter and Density Plots
The plotScatterMatrix function creates scatter plots for numerical columns and includes kernel density plots on the diagonal, aiding in understanding the joint distribution of variables.

Function Signature:
plotScatterMatrix(df, plotSize, textSize)

# Example usage:
plotScatterMatrix(df1, 10, 12)

Data Loading
The script includes an example of loading a dataset (mbti_1.csv). Adjust the file path and name as needed for your specific dataset.

nRowsRead = 1000  # specify 'None' if you want to read the entire file
df1 = pd.read_csv('mbti_1.csv', delimiter=',', nrows=nRowsRead)
df1.dataframeName = 'mbti_1.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)

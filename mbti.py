from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import os  # For accessing the directory structure
import pandas as pd  # For data processing, including reading CSV files

# Define a function to create distribution graphs (histograms or bar plots) for columns in a DataFrame.
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()  # Calculate the number of unique values in each column.
    # Filter columns for displaying, selecting those with 1 to 50 unique values.
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape  # Get the number of rows and columns in the filtered DataFrame.
    columnNames = list(df)  # Create a list of column names from the filtered DataFrame.
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # Calculate the number of rows for graph arrangement.

    # Create a new figure for plotting with specified dimensions and properties.
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')

    for i in range(min(nCol, nGraphShown)):  # Loop through columns up to the specified number of graphs to show.
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)  # Create subplots within the figure.

        columnDf = df.iloc[:, i]  # Select the data from the current column.

        if not np.issubdtype(type(columnDf.iloc[0]), np.number):  # Check if the data is non-numeric.
            valueCounts = columnDf.value_counts()  # Calculate value counts for each category.
            valueCounts.plot.bar()  # Create a bar plot to visualize the distribution of categories.
        else:
            columnDf.hist()  # If the data is numeric, create a histogram to visualize the distribution.

        plt.ylabel('counts')  # Add a label to the y-axis.
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability.
        plt.title(f'{columnNames[i]} (column {i})')  # Set the title for each graph.

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)  # Ensure well-arranged plots with proper padding.
    plt.show()  # Display the plots.

# The function can be called with the following parameters:
# plotPerColumnDistribution(data_frame, number_of_graphs_to_show, graphs_per_row)
# Example usage:
# plotPerColumnDistribution(df, 12, 4)

# Define a function to plot the correlation matrix.
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName  # Store the name of the DataFrame in the 'filename' variable.
    df = df.dropna('columns')  # Drop columns containing NaN (missing) values.

    # Keep only columns with more than one unique value (non-constant columns).
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return  # If there are less than two columns to plot, return and display a message.

    corr = df.corr()  # Calculate the correlation matrix.

    # Create a figure for plotting with specified dimensions and properties.
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    # Create a matrix plot (heatmap) of the correlation matrix.
    corrMat = plt.matshow(corr, fignum=1)

    # Set x-axis labels and rotate them for better readability.
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    # Set y-axis labels.
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Move the x-axis ticks to the bottom for better visibility.
    plt.gca().xaxis.tick_bottom()

    # Add a color bar to the plot to represent the correlation values.
    plt.colorbar(corrMat)

    # Set the title for the plot, including the filename and fontsize.
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    # Display the correlation matrix plot.
    plt.show()

# The function can be called with the following parameters:
# plotCorrelationMatrix(data_frame, graph_width)
# Example usage:
# plotCorrelationMatrix(df, 8)

# Define a function to plot scatter and density plots for numerical columns.
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # Keep only numerical columns.

    # Remove rows and columns that would lead to df being singular (columns with NaN values).
    df = df.dropna('columns')

    # Keep only columns with more than one unique value (non-constant columns).
    df = df[[col for col in df if df[col].nunique() > 1]]

    columnNames = list(df)  # Create a list of column names from the DataFrame.

    # If there are more than 10 columns, limit to the first 10 columns for matrix inversion.
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]

    # Create a scatter matrix with kernel density plots on the diagonal.
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    # Annotate each subplot with the correlation coefficient.
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')  # Set a title for the entire plot.
    plt.show()

# The function can be called with the following parameters:
# plotScatterMatrix(data_frame, plot_size, text_size)
# Example usage:
# plotScatterMatrix(df, 10, 12)

nRowsRead = 1000 # specify 'None' if want to read whole file
# mbti_1.csv has 8675 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('mbti_1.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'mbti_1.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

df1.head(5)

plotPerColumnDistribution(df1, 10, 5)
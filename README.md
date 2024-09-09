# Cipla Stock Price Prediction

This project predicts Cipla's stock price using several machine learning models: LSTM, SVM, SVR, KNN, and K-Means Clustering. The project includes exploratory data analysis (EDA), data preprocessing, model training, and evaluation. Each model predicts the stock price, and results are compared with actual data using various performance metrics.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Implementation](#model-implementation)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
  - [LSTM Model](#lstm-model)
  - [SVM Model](#svm-model)
  - [SVR Model](#svr-model)
  - [KNN Model](#knn-model)
  - [K-Means Clustering](#k-means-clustering)
- [Evaluation and Stock Analysis](#evaluation-and-stock-analysis)
- [Results](#results)
- [License](#license)

## Project Overview

In this project, several models are trained to predict Cipla's stock price based on historical stock data. The models used include:
- **LSTM (Long Short-Term Memory)**
- **SVM (Support Vector Machine)**
- **SVR (Support Vector Regression)**
- **KNN (K-Nearest Neighbors)**
- **K-Means Clustering**

Each model is evaluated using RMSE (Root Mean Squared Error) and R² metrics, and predictions are plotted alongside the actual stock prices for visual comparison.

## Dataset

The dataset used contains the historical stock prices of Cipla. The main features include:
- **Date**: The date of the stock entry
- **Open**: Opening stock price for the day
- **High**: Highest stock price for the day
- **Low**: Lowest stock price for the day
- **Close**: Closing stock price for the day
- **Volume**: Number of shares traded

The dataset must be loaded as a CSV file named `CIPLA.csv`.

## Prerequisites

The following Python libraries are required to run this project:
```
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- tensorflow

```

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/cipla-stock-prediction.git

```

2. Navigate to the project directory:

```
cd cipla-stock-prediction

```

3. Place the dataset (```CIPLA.csv```) in the project folder.

4. Run the code file (```stock_prediction.py```).

## Model Implementation

## Exploratory Data Analysis (EDA)

Before training the models, EDA is performed to understand the data distribution and check for any anomalies or missing values.

- Plotting the **closing price** over time.
- Generating a **correlation heatmap** to visualize relationships between features.

## Data Preprocessing

- **Scaling**: The feature values (```Open```, ```High```, ```Low```, ```Volume```) are scaled using ```StandardScaler``` and ```MinMaxScaler```.

- **Splitting**: The dataset is split into training and testing sets using an 80-20 ratio.

## LSTM Model

The LSTM model is used to predict stock prices based on temporal data. The key steps include:

- Reshaping data for LSTM input format.

- Building a sequential LSTM network with multiple layers.

- Training the model for 50 epochs and generating predictions.

- Plotting the predicted vs. actual stock prices.

## SVM Model

Support Vector Machines (SVM) are used for regression. The steps include:

- Fitting the SVM model using a radial basis function (RBF) kernel.

- Generating predictions and plotting the results.

## SVR Model

Support Vector Regression (SVR) is another regression model, similar to SVM but specifically for continuous data. The implementation follows:

- Fitting the SVR model and making predictions.

- Visualizing predicted vs. actual stock prices.

## KNN Model

K-Nearest Neighbors (KNN) is a non-parametric method used for regression. Steps include:

- Fitting the KNN model using 5 neighbors.

- Predicting stock prices and comparing them to the actual values.

## K-Means Clustering

K-Means Clustering is used to group the stock data into clusters for further analysis. The steps are:

- Fitting the KMeans model and assigning clusters.

- Visualizing clusters based on stock volume and closing prices.

## Evaluation and Stock Analysis

Each model is evaluated using:

- **RMSE (Root Mean Squared Error)**: A standard measure of the model's prediction error.

- **R² (R-squared)**: Indicates how well the model fits the data.

## Evaluation Metrics

For each model, the following performance metrics are calculated:

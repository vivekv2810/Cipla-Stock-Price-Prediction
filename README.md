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

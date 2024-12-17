# INTELIMARK_AI
# Store Sales Prediction

This repository contains the solution to a sales prediction problem for a retail store. The goal is to predict the sales (`Item_Outlet_Sales`) of items in various outlets based on various features, such as item characteristics, outlet information, and item visibility. The solution uses data preprocessing, feature engineering, correlation analysis, and machine learning models to accurately predict the sales.

## Problem Overview

The task is to predict `Item_Outlet_Sales` using the given dataset, which contains various features related to items and outlets. Some of the columns include `Item_Weight`, `Item_Type`, `Outlet_Size`, and `Item_MRP`. The dataset also contains missing values, categorical variables, and potential correlations between features that need to be handled carefully for accurate predictions.

## Approach

### 1. **Data Preprocessing**

- **Handling Missing Values**:  
  Missing values in the `Item_Weight` column are imputed with the mean of the column. Missing values in the `Outlet_Size` column are filled with the most frequent value (mode).

- **Feature Engineering**:  
  A new feature, `Outlet_Age`, is created by subtracting the `Outlet_Establishment_Year` from the current year (2024). This new feature represents the age of each outlet, which can affect sales.

- **One-Hot Encoding**:  
  Categorical variables such as `Item_Type`, `Item_Fat_Content`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type` are transformed into numerical features using one-hot encoding to make them compatible with machine learning algorithms.

### 2. **Correlation Analysis**

A correlation matrix is generated to assess the relationship between each feature and the target variable (`Item_Outlet_Sales`). Features with low correlation (less than 0.1 or greater than -0.1) with the target variable are dropped from the model to avoid overfitting and improve model performance.

### 3. **Model Selection and Training**

- **Model Choice**:  
  The model used is `XGBoost Regressor` due to its robust performance on structured data and its ability to handle both numerical and categorical features effectively.

- **Hyperparameter Tuning**:  
  A grid search is performed to find the best hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, etc.) for the `XGBoost` model to optimize its performance.

- **Training**:  
  The model is trained using the optimal hyperparameters on the training dataset, and performance is validated on a separate validation set.

### 4. **Model Evaluation**

The model's performance is evaluated on the validation set using **Root Mean Squared Error (RMSE)**. This metric helps assess the accuracy of the predictions. The model is also tested on the test set, ensuring that all predictions are positive (sales cannot be negative).

### 5. **Submission Preparation**

A CSV file containing the `Item_Identifier`, `Outlet_Identifier`, and the predicted `Item_Outlet_Sales` for the test set is created and saved for submission.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Installation

To set up the environment, use the following commands:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/store-sales-prediction.git
   cd store-sales-prediction


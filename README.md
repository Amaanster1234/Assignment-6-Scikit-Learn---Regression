# Scikit-Learn Regression with the Diabetes Dataset

## Project Overview

The purpose of this project is to practice performing machine learning regression using Python and the Scikit-Learn library. The built-in diabetes dataset provided by Scikit-Learn is used to train and evaluate several regression models that attempt to predict disease progression based on medical measurements.

This project compares multiple regression models and evaluates their performance using several regression metrics. The goal is to determine which model provides the most accurate predictions.

## Dataset

This project uses the **Diabetes Dataset** included in Scikit-Learn.

The dataset contains:

* 442 samples
* 10 numerical input features
* 1 target variable

The features represent different medical measurements, such as age, body mass index (BMI), and blood serum measurements. The target variable represents a quantitative measure of disease progression one year after baseline.

## Data Splitting

The dataset was divided into three groups using the standard machine learning workflow:

* Training Set (60%) – Used to train the models
* Validation Set (20%) – Used to compare model performance and tune parameters
* Test Set (20%) – Used for final model evaluation

This process helps ensure that the model is evaluated on data it has not seen before.

## Regression Models Implemented

Three different regression models were built and evaluated.

### Linear Regression

Linear Regression attempts to model the relationship between the features and the target variable using a linear equation.

This model serves as a baseline model because it is simple and commonly used in regression tasks.

### K-Nearest Neighbors Regressor

The KNN Regressor predicts values based on the average of the closest data points in the training set.

Multiple values of **k** were tested:

* k = 3
* k = 5
* k = 7
* k = 9
* k = 11
* k = 13
* k = 15

The value that produced the lowest validation error was selected.

### Decision Tree Regressor

Decision Trees split the dataset into smaller regions based on feature values and make predictions using the average value in each region.

Several maximum depths were tested:

* depth = 2
* depth = 3
* depth = 4
* depth = 5
* depth = 6
* depth = 7
* depth = 8

The depth with the best validation performance was selected.

## Evaluation Metrics

Each model was evaluated using several regression metrics.

Mean Absolute Error (MAE)
Measures the average absolute difference between predicted and actual values.

Mean Squared Error (MSE)
Measures the average squared difference between predicted and actual values.

Root Mean Squared Error (RMSE)
The square root of MSE. This is commonly used because it is in the same units as the target variable.

R² (Coefficient of Determination)
Measures how well the model explains the variance in the data. Higher values indicate better performance.

The model with the lowest RMSE on the validation set was selected as the best model.

## Final Model Selection

After evaluating all models on the validation dataset, the model with the lowest RMSE was chosen as the best performing model.

The final model was then retrained using the combined training and validation datasets, and its performance was evaluated on the test dataset to measure how well it generalizes to unseen data.

## How to Run the Program

1. Install required libraries if they are not already installed:

pip install scikit-learn numpy

2. Run the program:

python main.py

3. The program will display:

* Dataset information
* Performance of each model
* Best performing model
* Final test results
* A short conclusion

## Project Structure

.
├── main.py
├── README.md

main.py – Python script that trains and evaluates the regression models
README.md – Explanation of the project and implementation

## Limitations

Some limitations of this project include:

* Only three regression models were tested
* Hyperparameter tuning was limited to a small set of values
* The dataset is relatively small compared to many real-world datasets
* No feature engineering or scaling techniques were applied

Future improvements could include testing additional models such as Random Forest or Support Vector Regression, performing more extensive hyperparameter tuning, or applying feature preprocessing techniques.

## Conclusion

This project demonstrates how machine learning regression models can be applied to medical data to predict disease progression. By comparing multiple models and evaluating their performance using several regression metrics, it is possible to determine which approach provides the most accurate predictions for this dataset.


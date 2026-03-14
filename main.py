"""
Assignment 6: Scikit Learn Regression

This program uses the built in diabetes dataset from scikit-learn
to train and compare three regression models:
1. Linear Regression
2. K-Nearest Neighbors Regressor
3. Decision Tree Regressor
"""

from math import sqrt

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def load_data():
    #Load the diabetes dataset and return features, targets, and feature names.
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    return X, y, feature_names

def split_data(X, y):
    """
    Split the data into training, validation, and test sets.
    60% training, 20% validation, 20% test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_regression(y_true, y_pred):
    #Calculate and return common regression metrics.
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def print_metrics(metrics):
    #Print regression metrics in a clean format.
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")
    print()

def train_linear_regression(X_train, y_train, X_val, y_val):
    #Train and evaluate a Linear Regression Model. 
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    metrics = evaluate_regression(y_val, predictions)

    return {
        "name": "Linear Regression",
        "model": model,
        "params": "default",
        "val_metrics": metrics
    }

def train_knn_models(X_train, y_train, X_val, y_val):
    #Train and evaluate several KNN models, then return the best one.
    best_result = None
    k_values = [3, 5, 7, 9, 11, 13, 15]

    print("Testing KNN Regressor models")
    print("=" * 40)

    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        metrics = evaluate_regression(y_val, predictions)

        result = {
            "name": "KNN Regressor",
            "model": model,
            "params": f"k={k}",
            "val_metrics": metrics 
        }

        print(f"KNN Regressor (k={k})")
        print_metrics(metrics)

        if best_result is None or metrics["RMSE"] < best_result["val_metrics"]["RMSE"]:
            best_result = result
    
    return best_result

def train_decision_tree_models(X_train, y_train, X_val, y_val):
    #Train and evaluate several Decision Tree models, then return the best one.
    best_result = None
    depth_values = [2, 3, 4, 5, 6, 7, 8]

    print("Testing Decision Tree Regressor models")
    print("=" * 40)

    for depth in depth_values:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        metrics = evaluate_regression(y_val, predictions)

        result = {
            "name": "Decision Tree Regressor",
            "model": model,
            "params": f"max_depth={depth}",
            "val_metrics": metrics
        }

        print(f"Decision Tree Regressor (max_depth={depth})")
        print_metrics(metrics)

        if best_result is None or metrics["RMSE"] < best_result["val_metrics"]["RMSE"]:
            best_result = result
    
    return best_result

def choose_best_model(results):
    #Choose overall best model based on lowest validation RSME.
    best_result = results[0]

    for result in results[1:]:
        if result["val_metrics"]["RMSE"] < best_result["val_metrics"]["RMSE"]:
            best_result = result
    
    return best_result

def retrain_best_model(best_result, X_train, X_val, y_train, y_val):
    #Retrain the best model using both training and validation data.
    X_full_train = np.concatenate((X_train, X_val), axis=0)
    y_full_train = np.concatenate((y_train, y_val), axis=0)

    if best_result["name"] == "Linear Regression":
        final_model = LinearRegression()
    
    elif best_result["name"] == "KNN Regressor":
        k = int(best_result["params"].split("1")[1])
        final_model = KNeighborsRegressor(n_neighbors=k)
    
    elif best_result["name"] == "Decision Tree Regressor":
        depth = int(best_result["params"].split("=")[1])
        final_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    
    else:
        raise ValueError("Unknown Model Type.")
    
    final_model.fit(X_full_train, y_full_train)
    return final_model

def main():
    #Run the full regression modeling process.
    X, y, feature_names = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Diabetes Regression Results")
    print("-" * 40)
    print(f"Number of samples:  {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names:", feature_names, "\n")
    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}", "\n")

    print("Testing Linear Regression")
    print("=" * 40)
    linear_result = train_linear_regression(X_train, y_train, X_val, y_val)
    print("Linear Regression (default)")
    print_metrics(linear_result["val_metrics"])

    knn_result = train_knn_models(X_train, y_train, X_val, y_val)
    tree_result = train_decision_tree_models(X_train, y_train, X_val, y_val)

    results = [linear_result, knn_result, tree_result]

    print("Best Version of Each Model on Validation Set")
    print("=" * 40)
    for result in results:
        print(f"{result['name']} ({result['params']})")
        print_metrics(result["val_metrics"])

    best_result = choose_best_model(results)

    print("Overall Best Model")
    print("=" * 40)
    print(f"Model:  {best_result['name']}")
    print(f"Params: {best_result['params']}")
    print_metrics(best_result["val_metrics"])

    final_model = retrain_best_model(best_result, X_train, X_val, y_train, y_val)
    test_predicions = final_model.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_predicions)

    print("Final Test Performance")
    print("=" * 40)
    print(f"Best model tested: {best_result['name']} ({best_result['params']})")
    print_metrics(test_metrics)

    print("Conclusion")
    print("=" * 40)
    print(
        f"The best model was {best_result['name']} with {best_result['params']}. "
        f"It had the lowest validation RMSE of {best_result['val_metrics']['RMSE']:.4f} "
        f"and an R2 value of {best_result['val_metrics']['R2']:.4f}, which made it "
        f"the strongest overall     model on the validation set. I also compared MAE and "
        f"MSE to support the final decision"
    )

if __name__ == "__main__":
    main()






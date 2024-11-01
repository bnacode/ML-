import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from common import *
from RandomForest import make_confusion_matrix



def find_best_degree(X, y):
    degrees = [1, 2, 3, 4, 5]

    # Create a parameter grid with different degrees
    param_grid = {'polynomialfeatures__degree': degrees}

    # Create a pipeline explicitly and include PolynomialFeatures and LinearRegression
    poly_reg = make_pipeline(PolynomialFeatures(), LinearRegression())

    # Create a GridSearchCV instance
    grid_search = GridSearchCV(poly_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X, y)
    best_degree = grid_search.best_params_['polynomialfeatures__degree']
    print(best_degree)
    return best_degree

def polynomial_regression(X, Y):
    X = config_C(X)
    best_degree = find_best_degree(X, Y)

    classifier = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())
    # poly_reg_best.fit(X, Y)

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='neg_mean_squared_error')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean squared error:", -scores.mean())


def linear_regression(X, Y):
    # X, Y = normalize_continual_values(X, Y)

    # X = config_B(X)

    # Use the best hyperparameters to train the final SVM model
    classifier = LinearRegression()

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='neg_mean_squared_error')



    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean squared error:", -scores.mean())



def logistic_regression(X, Y):
    # X, Y = change_to_class_values(X, Y)
    X, Y = change_to_binary_values(X, Y)

    # X = config_B(X)

    classifier = LogisticRegression(max_iter=5000)


    # make_confusion_matrix(X, Y, classifier)

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='accuracy')

    make_confusion_matrix(X, Y, classifier)

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean()*100)




if __name__ == '__main__':
    X, Y = init()
    logistic_regression(X, Y)
    # linear_regression(X, Y)
    # polynomial_regression(X, Y)





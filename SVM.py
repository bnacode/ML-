from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR

from common import config_C, config_B, init, change_to_class_values, change_to_binary_values


def find_best_param(X_train, Y_train, model):

    # Define the parameter grid for SVM with RBF kernel
    param_grid = {'gamma': [2 ** (-9), 2 ** (-7), 2 ** (-5), 2 ** (-3), 2 ** (-1)]}
    print(param_grid)
    if model == 'regression':
        svm_model = SVR(kernel='rbf')
    else:
        svm_model = SVC(kernel='rbf')

    # Instantiate the GridSearchCV object
    grid_search = GridSearchCV(svm_model, param_grid, cv=10, n_jobs=-1)

    # Fit the grid search to the data (use only training data for internal grid search)
    grid_search.fit(X_train, Y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(best_params)
    return best_params


def SVM_classification(X, Y):
    # X, Y = change_to_class_values(X, Y)
    X, Y = change_to_binary_values(X, Y)

    # X = config_B(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    best_params = find_best_param(X_train, Y_train, 'classification')

    # Use the best hyperparameters to train the final SVM model
    classifier = SVC(kernel='rbf', **best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean() * 100)




def SVM_regression(X, Y):
    X = config_B(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    best_params = find_best_param(X_train, Y_train, 'regression')

    # Use the best hyperparameters to train the final SVM model
    classifier = SVC(kernel='rbf', **best_params)

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='neg_root_mean_squared_error')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", -scores.mean())


if __name__ == '__main__':
    X, Y = init()
    SVM_classification(X, Y)
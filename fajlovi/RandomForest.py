from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import common
from common import *
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


def calculate_feature_importance(classifier):

    feature_names = common.feature_names

    print("Feature Names:", feature_names)

    # Get feature importances
    importances = classifier.feature_importances_

    # Create a dictionary mapping feature names to their importance scores
    feature_importance_dict = dict(zip(feature_names, importances))

    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract sorted feature names and importances
    sorted_feature_names, sorted_importances = zip(*sorted_features)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), sorted_importances, align="center")
    plt.xticks(range(len(feature_names)), sorted_feature_names, rotation=90,
               ha='right')  # Rotate x-tick labels for better readability
    plt.xlabel("Feature Name")
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()

def make_confusion_matrix(X, Y, classifier):

    X_train, X_test, Y_train, Y_test = train_and_test_slit(X, Y)
    classifier.fit(X_train, Y_train)

    predicted_values = classifier.predict(X)


    conf_matrix = confusion_matrix(Y, predicted_values)

    class_labels = ['Fail', 'Pass']
    # class_labels = ['A', 'B', 'C', 'D', 'F']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



def calculate_RF_for_classifier(X, Y):

    X, Y = change_to_class_values(X, Y)
    # X, Y = change_to_binary_values(X, Y)

    # X = config_B(X)

    scoring = 'accuracy'
    classifier = RandomForestClassifier(n_estimators=500, random_state=42)



    # Define the cross-validation strategy: 10-fold cross-validation, repeated 20 times
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)

    scores = cross_val_score(classifier, X, Y, cv=cv, scoring=scoring)

    make_confusion_matrix(X, Y, classifier)

    calculate_feature_importance(classifier)
    print("Mean Accuracy:", scores.mean()*100)


def calculate_RF_for_regressor(X, Y):
    # X, Y = normalize_continual_values(X, Y)

    # X = config_C(X)

    scoring = 'neg_root_mean_squared_error'
    reggresor = RandomForestRegressor(n_estimators=500, random_state=42)

    # Define the cross-validation strategy: 10-fold cross-validation, repeated 20 times
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)
    scores = cross_val_score(reggresor, X, Y, cv=cv, scoring=scoring)

    # Print the mean and standard deviation of the cross-validation scores for coninual values


    print("Mean Accuracy:", -scores.mean())
    print("Standard Deviation of Accuracy:", scores.std())



def calculate_DT_for_classifier(X, Y, best_param = None):
    # X, Y = change_to_class_values(X, Y)
    # X, Y = change_to_binary_values(X, Y)

    X = config_C(X)

    scoring = 'accuracy'
    classifier = DecisionTreeClassifier(**best_param)


    # Define the cross-validation strategy: 10-fold cross-validation, repeated 20 times
    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring=scoring)

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean()*100)


def calculate_DT_for_regressor(X, Y, best_param = None):

    scoring = 'neg_root_mean_squared_error'
    classifier = DecisionTreeRegressor(**best_param)

    # Define the cross-validation strategy: 10-fold cross-validation, repeated 20 times
    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring=scoring)

    # Print the mean and standard deviation of the cross-validation scores for coninual values
    print("Mean Squared Error:", -scores.mean())



def find_best_param_for_DT(X, Y):
    # X, Y = change_to_class_values(X, Y)
    # X, Y = change_to_binary_values(X, Y)


    # X = config_C(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
        # 'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [5, 10, 20, 30, 50, 70, 100],
        'min_samples_leaf': [1, 2, 4, 5, 10, 20]
    }

    # Create the Decision Tree classifier
    scoring = 'neg_root_mean_squared_error'
    classifier = DecisionTreeRegressor()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring=scoring)

    # Perform the grid search on the training data
    grid_search.fit(X_train, Y_train)

    # Print the best parameters found
    best_params = grid_search.best_params_

    return best_params


    # scoring = 'accuracy'
    # classifier = DecisionTreeClassifier()










if __name__ == '__main__':
    X, Y = init()
    # best_param = find_best_param_for_DT(X.copy(), Y.copy())
    # calculate_RF_for_regressor(X, Y)
    #
    # X = config_B(X)
    #
    # classifier = RandomForestClassifier()
    # classifier.fit(X, Y)
    # calculate_feature_importance(classifier)

    reggresor = RandomForestRegressor()
    reggresor.fit(X, Y)
    y_pred = reggresor.predict(X)
    y_true = Y

    # Create scatter plot for predicted values in blue
    plt.scatter(y_true, y_pred, color='blue', label='Predicted Values')

    # Create scatter plot for true values in orange
    plt.scatter(y_true, y_true, color='orange', label='True Values')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Predicted vs. Real Values')
    plt.legend()
    plt.show()


















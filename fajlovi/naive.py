import numpy as np
from matplotlib import pyplot as plt

from RandomForest import make_confusion_matrix
from common import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

continuous_feature_names = ['age','absences']
binary_feature_names = ['school', 'address', 'famsize', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
                        'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other',
                        'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home',
                        'reason_other', 'reason_reputation', 'guardian_father',
                        'guardian_mother', 'guardian_other', 'schoolsup', 'famsup', 'paid',
                        'activities', 'nursery', 'higher', 'internet', 'romantic']
multiclass_feature_names = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                            'freetime', 'goout', 'Dalc', 'Walc', 'health']

def predict_B_config_predict(X):
    return X[:, -2]


def predict_A_config_predict(X):
    return X[:, -1]

def predict_C_config_binary_or_class_predict(X, Y_train):

    bin_counts = np.bincount(Y_train)
    most_common_class = np.argmax(bin_counts)

    return np.full(X.shape[0], most_common_class)

def predict_C_config_continual_predict(X, Y_train):
    mean = np.mean(Y_train)
    return np.full(X.shape[0], mean)

def get_continuous_features():
    continuous_features = ['absences', 'G1']

def majority_voting(predictions):
    # Transpose the array to get predictions for each sample in columns
    transposed_predictions = np.array(predictions).T
    # Use np.apply_along_axis to apply np.bincount to each row
    combined_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1,
                                               arr=transposed_predictions)
    return combined_predictions

def GNB(X, Y):
    Y = calculate_binary_values_array(Y)
    continuous_features = get_indices(continuous_feature_names)

    classifier = GaussianNB()

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X[:, continuous_features], Y, cv=cv, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean() * 100)

def MNB(X, Y):
    Y = calculate_binary_values_array(Y)
    multiclass_feature = get_indices(multiclass_feature_names)

    classifier = MultinomialNB()

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X[:, multiclass_feature], Y, cv=cv, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean() * 100)

def BNB(X, Y):
    Y = calculate_binary_values_array(Y)
    binary_feature = get_indices(binary_feature_names)

    classifier = BernoulliNB()

    cv = RepeatedKFold(n_splits=10, n_repeats=20)

    # Perform cross-validation
    scores = cross_val_score(classifier, X[:, binary_feature], Y, cv=cv, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean() * 100)

def NB(X, y):
    y = calculate_class_values_array(y)
    # X = config_B(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Separate features based on their types
    continuous_features = get_indices(continuous_feature_names)
    binary_features = get_indices(binary_feature_names)
    multiclass_features = get_indices(multiclass_feature_names)

    # Train the Naive Bayes classifiers on their respective feature subsets
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()

    gnb.fit(X_train[:, continuous_features], y_train)
    mnb.fit(X_train[:, multiclass_features], y_train)
    bnb.fit(X_train[:, binary_features], y_train)

    # Make predictions on the test data
    y_pred_gnb = gnb.predict(X_test[:, continuous_features])
    y_pred_mnb = mnb.predict(X_test[:, multiclass_features])
    y_pred_bnb = bnb.predict(X_test[:, binary_features])

    combined_predictions = majority_voting([y_pred_gnb, y_pred_mnb, y_pred_bnb])

    # Evaluate the combined predictions
    print("Combined", accuracy_score(y_test, combined_predictions) * 100)
    print(accuracy_score(y_pred_gnb, y_test) * 100)
    print(accuracy_score(y_pred_mnb, y_test) * 100)
    print(accuracy_score(y_pred_bnb, y_test) * 100)

    accuracy_gnb = accuracy_score(y_pred_gnb, y_test)
    accuracy_mnb = accuracy_score(y_pred_mnb, y_test)
    accuracy_bnb = accuracy_score(y_pred_bnb, y_test)
    accuracy_combined = accuracy_score(combined_predictions, y_test)

    print("Max:", np.max([accuracy_gnb, accuracy_bnb, accuracy_mnb, accuracy_combined]) * 100)


if __name__ == '__main__':
    X, Y = init()

    # X, Y = change_to_class_values(X, Y)
    X, Y = change_to_binary_values(X, Y)
    # X, Y = normalize_continual_values(X, Y)

    # X_train, X_test, Y_train, Y_test = train_and_test_slit(X, Y)

    Y_pred = predict_A_config_predict(X)
    # Y_pred = predict_C_config_continual_predict(X, Y)
    # Y_pred = predict_B_config_predict(X)
    accuracy = accuracy_score(Y_pred, Y)
    print("Accuracy:", accuracy * 100)

    # Y_pred = predict_C_config_continual_predict(X_test, Y_train)
    # rmse = np.sqrt(mean_squared_error(Y_pred, Y))
    # print(rmse)


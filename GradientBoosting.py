from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score

from common import *


def gradient_boosting_classification(X, Y):
    X, Y = change_to_class_values(X, Y)
    # X, Y = change_to_binary_values(X, Y)
    # X = config_B(X)

    classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='accuracy')

    # Print the mean and standard deviation of the cross-validation scores for class values
    print("Mean Accuracy:", scores.mean() * 100)

if __name__ == '__main__':
    X, Y = init()
    gradient_boosting_classification(X, Y)


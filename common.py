import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

file_path = '../data/student+performance/student/student-por.csv'


feature_names = None

def load_data_frame():
    data = pd.read_csv(file_path, header=0, sep=';')
    return data

def get_feature_names():
    return feature_names

def init():
    global file_path

    data = pd.read_csv(file_path, header=0, sep=';')
    data = mapping(data)

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    return X, Y


def train_and_test_slit(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    return X_train, X_test, Y_train, Y_test

def mapping(data):
    global feature_names

    # Mapping for 'sex'
    sex_mapping = {'F': 0, 'M': 1}
    data['sex'] = data['sex'].map(sex_mapping)

    # Mapping for 'school'
    school_mapping = {'GP': 0, 'MS': 1}
    data['school'] = data['school'].map(school_mapping)

    # Mapping for 'address'
    address_mapping = {'U': 0, 'R': 1}
    data['address'] = data['address'].map(address_mapping)

    # Mapping for 'Pstatus'
    pstatus_mapping = {'T': 0, 'A': 1}
    data['Pstatus'] = data['Pstatus'].map(pstatus_mapping)

    # Mapping for 'famsize'
    famsize_mapping = {'LE3': 0, 'GT3': 1}
    data['famsize'] = data['famsize'].map(famsize_mapping)

    # Yes and No mapping
    yes_or_no_mapping = {'no': 0, 'yes': 1}
    columns_to_map = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    data[columns_to_map] = data[columns_to_map].apply(lambda col: col.map(yes_or_no_mapping))

    #One hot encoding
    categorical_columns = ['Mjob', 'Fjob', 'reason', 'guardian']

    # Apply one-hot encoding to the specified columns
    one_hot_encoded = pd.get_dummies(data[categorical_columns], prefix=categorical_columns)
    data = pd.concat([data.iloc[:, :-3], one_hot_encoded, data.iloc[:, -3:]], axis=1)
    data.drop(columns=categorical_columns, axis=1, inplace=True)
    data = data.astype(int)

    feature_names = data.columns[:-1]


    return data


def config_B(X):
    X_without_G2 = X[:, :-1]
    return X_without_G2

def config_C(X):
    X_without_G1_and_G2 = X[:, :-2]
    return X_without_G1_and_G2


def normalize_array(array):
    # scaler = StandardScaler()
    # return scaler.fit_transform(array.reshape(-1, 1)).flatten()
    array = (array - np.mean(array)) / np.std(array)
    return array


def calculate_class_values_array(array):
    score_classes = {
        'A': range(16, 21),
        'B': range(14, 16),
        'C': range(12, 14),
        'D': range(10, 12),
        'F': range(0, 10),
    }

    # Assuming 'Y' is your original array with grades
    array_mapped = np.array(
        [i for grade in array for i, (category, scores) in enumerate(score_classes.items()) if grade in scores])

    return array_mapped

def calculate_binary_values_array(array):
    array_binary = (array >= 10).astype(int)
    return array_binary


def change_to_class_values(X, Y):
    X[:, -2] = calculate_class_values_array(X[:, -2])
    X[:, -1] = calculate_class_values_array(X[:, -1])
    Y = calculate_class_values_array(Y)
    return X, Y

def change_to_binary_values(X, Y):
    # X[:, -2] = calculate_binary_values_array(X[:, -2])
    # X[:, -1] = calculate_binary_values_array(X[:, -1])
    Y = calculate_binary_values_array(Y)
    return X, Y

def normalize_continual_values(X, Y):
    X[:, 2] = normalize_array(X[:, 2])
    X[:, -2] = normalize_array(X[:, -2])
    X[:, -1] = normalize_array(X[:, -1])
    Y = normalize_array(Y)
    return X, Y


def get_indices(names):
    indicies = []
    features = get_feature_names().tolist()
    for name in names:
        indicies.append(features.index(name))
    return  indicies


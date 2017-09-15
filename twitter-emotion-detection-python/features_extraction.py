import csv
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def feature_selection(X, X2, y, features_names, file_valences, file_arousal, file_merged, file_importances):
    """
    Feature selection

    Selects top features
    :param X: Array with features for every data row
    :type X: (np.array)
    :param y: Labels for every data row
    :type y: (np.array)
    :param file_valences: file to save the selected features from X
    :type file_valences: (str)
    :return:

    """
    print("feature_selection")

    X = X[:y.shape[0]]

    X2 = X2[:y.shape[0]]

    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y[:, 0])

    # model = SelectFromModel(clf, prefit=True)
    # X_new = model.transform(X)
    mask = clf.feature_importances_ > 0.0006
    X_new = X[:, mask]
    X2_new = X2[:, mask]
    emotions = y[:, 0]
    features_names = features_names[mask]
    temp = pd.DataFrame(clf.feature_importances_[mask])
    temp['features_names'] = features_names
    temp.to_csv(file_importances, index=False)
    temp = pd.DataFrame(X_new)
    temp['emotion_class'] = emotions
    temp.to_csv(file_valences, index=False)
    valences = temp.copy()
    valences.columns = ['v_' + str(i) for i in range(np.sum(mask))] + ['emotion']
    temp = pd.DataFrame(X2_new)
    temp['emotion_class'] = emotions
    temp.to_csv(file_arousal, index=False)
    arousals = temp.copy()
    arousals.columns = ['a_' + str(i) for i in range(np.sum(mask))] + ['emotion']
    merged = pd.concat([valences, arousals], axis=1)
    merged = merged.ix[:, 0:merged.shape[1] - 1]
    merged.to_csv(file_merged, index=False)
    print(np.sum(mask))


def generate_initial_features(file, feature_words, file1, file2, file3, size):
    """

    :param file:
    :param feature_words:
    :return:
    """
    print("generate_initial_features")
    n = len(feature_words)
    df = pd.DataFrame(0.0, index=np.arange(size), columns=feature_words)
    df2 = pd.DataFrame(0.0, index=np.arange(size), columns=feature_words)
    ind = 0
    y = np.empty([size, 1], dtype='str')

    with open(file) as csvDataFile:
        csv_reader = csv.reader(csvDataFile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            # df.loc[-1] = np.zeros(n)            # adding a row
            # df.index = df.index + 1             # shifting index
            y[ind] = row[1]

            for i in range(2, len(row), 3):
                df.set_value(index=ind, col=row[i].lower(), value=float(row[i + 1]))
                df2.set_value(index=ind, col=row[i].lower(), value=float(row[i + 2]))

            ind += 1
            print(ind)
            # df = df.sort_index()

    df.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)
    temp = pd.DataFrame(y)
    temp.to_csv(file3, index=False)
    return df, df2, y


def load_all_feature_names(file):
    """
    Loads all the words as separate features

    :param file: The file with id, emotion, [lemma, valence, arousal] for every tweet
    :type file: str
    :return: list with every lemma (word) as keys and 0 as values
    :rtype: list
    """
    feature_names = {}
    with open(file) as csvDataFile:
        csv_reader = csv.reader(csvDataFile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            for i in range(2, len(row), 3):
                if row[i] not in feature_names.keys():
                    feature_names[row[i].lower()] = 0
    return feature_names.keys()


if __name__ == '__main__':
    words = load_all_feature_names('data/output.csv')
    file1 = 'data/valence_features.csv'
    file2 = 'data/arousal_features.csv'
    file3 = 'data/emotions.csv'
    if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3):
        x = pd.read_csv(file1)
        x2 = pd.read_csv(file2)
        y = pd.read_csv(file3).as_matrix()
    else:
        x, x2, y = generate_initial_features('data/output.csv', words, file1, file2, file3, 5348)

    X = x.as_matrix()
    X2 = x2.as_matrix()
    feature_selection(X, X2, y, x.columns.values,
                      'data_final/selected_features_valence.csv',
                      'data_final/selected_features_arousal.csv',
                      'data_final/selected_features_merged.csv',
                      'data/feature_importances.csv')

    print()

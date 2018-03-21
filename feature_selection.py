import numpy as np
from itertools import chain
from nltk.corpus import stopwords
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def feature_selection(X, y, feature_names):
    """ Feature selection

    Selects top features
    :param X: Array with features for every data row
    :type X: numpy.array
    :param y: Labels for every data row
    :type y: numpy.array
    :param feature_names: names of the features
    :type feature_names: list
    :return: selected features and the mask
    :rtype: (list, list)

    """
    print("feature_selection")
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)

    mask = clf.feature_importances_ > 0.0003
    selected = np.array(feature_names)[mask].tolist()
    print([(word, importance) for importance, word in
           sorted(zip(clf.feature_importances_[mask], selected), reverse=True)])

    return selected, mask


def generate_initial_features(dataset, vocab=None):
    """ Create the initial features from the vocabulary of the dataset.

    :param dataset: dataset containing the tweets with columns 'id', 'class', 'lemmas', and 'valences'
    :type dataset: pandas.DataFrame
    :param vocab: vocabulary of the dataset
    :type vocab: list
    :return: tuple of the dataset with initial features and the vocabulary
    :rtype: (pandas.DataFrame, list)
    """
    if vocab is None:
        vocab = get_vocabulary(dataset)
    # dictionary of every word in the vocabulary with its index
    dictionary = dict()
    for i in range(len(vocab)):
        dictionary[vocab[i]] = i

    # create initial feature for classification
    featured_dataset = dataset[['id', 'class', 'lemmas', 'valences']]
    for index, row in dataset.iterrows():
        # move the valences
        valences = np.array(list(chain(*row.valences))) + 10
        lemmas = list(chain(*row.lemmas))
        featured_dataset.set_value(index=index, col='lemmas', value=lemmas)
        # -1 is the value when the word is not present
        initial_valences = np.zeros(len(vocab))
        for lemma, valence in zip(lemmas, valences):
            initial_valences[dictionary[lemma.lower()]] = valence
        featured_dataset.set_value(index=index, col='valences', value=initial_valences)

    return featured_dataset, list(dictionary.keys())


def get_vocabulary(dataset):
    """ Get the vocabulary from the dataset

    :param dataset: The dataset containing the column 'lemmas'
    :type dataset: padnas.DataFrame
    :return: vocabulary
    :rtype: list
    """
    # vocabulary
    vocab = [item.lower() for lemmas in dataset.lemmas.values for item in list(chain(*lemmas))]
    vocab = np.unique(vocab)
    return vocab

import os
import keras as k
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import FeatureExtractionContextValenceShifting
from feature_selection import generate_initial_features, feature_selection
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, get_lemmas


def load_dataset(split):
    df = pd.read_csv('data/text_emotion.csv')
    df.columns = ['id', 'class', 'author', 'tweet']

    if os.path.exists('data_ml/text_emotion_features.npy'):
        X = np.load('data_ml/text_emotion_features.npy')
    else:
        print('Fix encoding...')
        df = fix_encoding(df)
        print('Split sentences...')
        df = split_tweet_sentences(df)
        print('Tokenize tweets...')
        df = tokenize_tweets(df)
        print('Lematize tweets...')
        df = get_lemmas(df)
        lexicon = pd.read_csv('lexicons/Ratings_Warriner_et_al.csv', usecols=[0, 1, 2, 5], index_col=0)
        lexicon.columns = ['word', 'valence', 'arousal']
        path_to_jar = 'stanford_parser/stanford-parser.jar'
        path_to_models_jar = 'stanford_parser/stanford-parser-3.9.1-models.jar'
        valence_shifter = FeatureExtractionContextValenceShifting(path_to_jar, path_to_models_jar, lexicon)
        df = valence_shifter.get_initial_valences(df)
        featured_dataset, vocab = generate_initial_features(df)
        X = featured_dataset['valences'].values.tolist()[:split]
        y = featured_dataset['class'].values.tolist()[:split]
        selected, mask = feature_selection(X, y, vocab)
        for index, row in featured_dataset.iterrows():
            valences = np.array(row.valences[mask])
            featured_dataset.set_value(index=index, col='valences', value=valences)
        X = np.vstack(featured_dataset.valences.values)
        np.save('data_ml/text_emotion_features', X)

    classes = df['class'].values.tolist()
    c = np.unique(classes).tolist()
    d = dict([(y, x) for x, y in enumerate(c)])
    classes = np.array([d[x] for x in classes])

    return X, classes, len(c)


def train_random_forest(split):
    data_X, data_y, n_classes = load_dataset(split)
    train_X = data_X[:split]
    train_y = data_y[:split]
    train_y = k.utils.to_categorical(train_y, n_classes)

    test_X = data_X[split:]
    test_y = data_y[split:]
    test_y = k.utils.to_categorical(test_y, n_classes)

    model = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                   bootstrap=True, n_jobs=-1)
    print('Training Random Forest model.')
    model.fit(train_X, train_y)
    print('Training Random Forest model done.')
    print('Testing Random Forest model.')
    predicted_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, predicted_y)
    print('Testing Random Forest model done.')
    print(accuracy)

    # filename = 'models_ml/random_forest_model.pkl'
    # joblib.dump(model, filename)


if __name__ == '__main__':
    train_random_forest(30000)

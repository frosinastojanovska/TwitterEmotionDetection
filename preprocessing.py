import numpy as np
import pandas as pd
from lemmatization import lemmatize
from lemmatization import pos_tagging
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import ftfy
from textblob import Word


def split_tweet_sentences(df):
    """ Gets dataset of tweets and split the sentences

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified column tweet in the data frame
    :rtype: pandas.DataFrame
    """
    df['tweet'] = [sent_tokenize(tweet) for tweet in df.tweet]
    return df


def tokenize_tweets(df):
    """ Gets dataset of tweets and tokenize each of them

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified data frame with one new column tokens
    :rtype: pandas.DataFrame
    """
    tknzr = TweetTokenizer()
    df['tokens'] = [[tknzr.tokenize(sent) for sent in text] for text in df.tweet]
    return df


def fix_encoding(df):
    """ Gets dataset of tweets text and fixes them

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified data frame tweet column with new fixed/encoded text
    :rtype: pandas.DataFrame
    """
    df.tweet = [ftfy.fix_text(tweet) for tweet in df.tweet]
    return df


def get_lemmas(df):
    """ Gets the data frame containing the dataset and finds the lemmas. Additionally here thw user tags are
    replaced with @user.

    :param df: data frame containing column tokens for tweet tokens
    :type df: pandas.DataFrame
    :return: modified data frame tokens column with added lemmas to the tokens in column lemmas
    :rtype: pandas.DataFrame
    """
    df['lemmas'] = ''
    for index, row in df.iterrows():
        tokens_lemmas = []
        for sent in row.tokens:
            tags = pos_tagging(sent)
            lemmas = lemmatize(tags)
            tokens_lemmas.append([x[1] for x in lemmas])

        df.set_value(index=index, col='lemmas', value=tokens_lemmas)

    return df


def get_word_embeddings(df):
    """ Gets the data frame containing the dataset and converts tweet tokens into corresponding word embeddings.

    :param df: data frame containing column tokens for tweet tokens
    :type df: pandas.DataFrame
    :return: modified data frame with new column for embedding representation of tweets
    :rtype: pandas.DataFrame
    """
    word_embeddings = load_embeddings('data/glove.twitter.27B.100d.txt')
    df['embeddings'] = ''
    for index, row in df.iterrows():
        embeddings = [encode_word(token, word_embeddings) for sent in row.tokens for token in sent]
        df.set_value(index=index, col='embeddings', value=embeddings)
    return df


def load_embeddings(file_name):
    """ Loads word embeddings from the given file

    :param file_name: name of the file containing word embeddings
    :type file_name: str
    :return: dictionary of words with their corresponding word embeddings
    :rtype: dict
    """
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = parts[1:]
            embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings


def encode_word(word, embeddings):
    """ Convert word to its word embedding vector. If the word is not contained in embeddings dictionary, word
    embedding of its corrected version is calculated. If the corrected word is not contained in embeddings
    dictionary, zero list is returned.

    :param word: word to be converted into its word embedding representation
    :type word: str
    :param embeddings: dictionary of words with their corresponding word embeddings
    :type embeddings: dict
    :return: word embedding representation of the word
    :rtype: list
    """
    word = word.lower()
    if word in embeddings.keys():
        vec = embeddings[word]
    else:
        w = Word(word)
        w = w.spellcheck()[0][0]
        if w in embeddings.keys():
            vec = embeddings[w]
        else:
            vec = [0] * 100
    return vec


if __name__ == '__main__':
    df = pd.read_excel('data/merged_datasets.xlsx')
    df = fix_encoding(df)
    df = tokenize_tweets(df)
    df = get_lemmas(df)
    df = df.drop(['emotion_intensity', 'tweet'], axis=1)
    df.to_csv("data/merged_datasets_tokens.csv", index=False)

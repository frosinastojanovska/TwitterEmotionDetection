import pandas as pd
from lemmatization import lemmatize
from lemmatization import pos_tagging
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
import ftfy


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


if __name__ == '__main__':
    df = pd.read_excel('data/merged_datasets.xlsx')
    df = fix_encoding(df)
    df = tokenize_tweets(df)
    df = get_lemmas(df)
    df = df.drop(['emotion_intensity', 'tweet'], axis=1)
    df.to_csv("data/merged_datasets_tokens.csv", index=False)

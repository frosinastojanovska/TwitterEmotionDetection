import pandas as pd
from lemmatization import lemmatize
from lemmatization import pos_tagging
from nltk.tokenize import TweetTokenizer
import ftfy


def tokenize_tweets(df):
    # tokenize tweet text to words
    tknzr = TweetTokenizer()
    df['tokens'] = [tknzr.tokenize(text) for text in df.tweet]
    return df


def fix_encoding(df):
    df.tweet = [ftfy.fix_text(tweet) for tweet in df.tweet]
    return df


def get_lemmas(df : pd.DataFrame):

    for index, row in df.iterrows():
        tags = pos_tagging(row.tokens)
        tokens_lemmas = lemmatize(tags)
        tokens_lemmas = ";".join(["//".join(tup) for tup in tokens_lemmas]).replace(',//,;', '').replace(",", ".").replace("////;", "").replace("////", "")
        df.set_value(index=index, col='tokens', value=tokens_lemmas)

    return df


if __name__ == '__main__':
    df = pd.read_excel('data/full_dataset.xlsx')
    df = fix_encoding(df)
    df = tokenize_tweets(df)
    df = get_lemmas(df)
    # df.tokens = df.tokens.apply(lambda x: ";".join("//".join([x[0], x[1]])).replace(',', ''))
    df = df.drop(['emotion_intensity', 'tweet'], axis=1)
    df.to_csv("data/full_dataset_tokens.csv", index=False)

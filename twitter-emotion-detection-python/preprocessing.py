import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
import ftfy


def try_tkenization():
    text = "When you've still got a whole season of Wentworth to watch and a stupid cunt in work ruins it for us :(( ­ @__KirstyGA #raging #oldcunt"

    # tokenize text into sentences
    res = nltk.sent_tokenize(text)
    print(res)

    # tokenize sentence to words
    res = nltk.word_tokenize(text)
    print(res)


def tokenize_tweets(df):
    # tokenize tweet text to words
    tknzr = TweetTokenizer()
    df['tokens'] = [tknzr.tokenize(text) for text in df.tweet]
    return df


def fix_encoding(df):
    df.tweet = [ftfy.fix_text(tweet) for tweet in df.tweet]
    return df


if __name__ == '__main__':
    df = pd.read_excel('data/full_dataset.xlsx')
    df = fix_encoding(df)
    df = tokenize_tweets(df)
    print(df.tokens.head(20))
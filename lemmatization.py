from nltk import WordNetLemmatizer
from nltk import pos_tag
from nltk import map_tag
import re


def pos_tagging(tweet):
    """ Finds the pos tag with nltk post_tag function, and then maps them with
    a tag required for the lemmatize function.

    :param tweet: list of words (tokens) represented like strings
    :type tweet: list
    :return: list of tuple (word, tag)
    :rtype: list
    """
    dict_tags = {'ADJ': 'a', 'ADJ_SAT': 's', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}

    tokens_tags = [(tokens[0], dict_tags[map_tag('en-ptb', 'universal', tokens[1])])
                   if map_tag('en-ptb', 'universal', tokens[1]) in dict_tags
                   else (tokens[0], '') for tokens in pos_tag(tweet)]

    return tokens_tags


def lemmatize(tweet):
    """ Finds lemma of words in tweets tagged with the required post tag
    that are mapped with the pos_tagging function: ['a', 's', 'r', 'n', 'v']

    :param tweet: list of tokens with post tagging tag
    :type tweet: list (string, string)
    :return: list of tuples (word, lemma)
    :rtype: list
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    tweet_list = [correct_word_and_get_lemma(wordnet_lemmatizer, tag) for tag in tweet]
    return tweet_list


def correct_word_and_get_lemma(wordnet_lemmatizer, token_tag):
    word = token_tag[0]
    pos_t = token_tag[1]

    # word = re.sub(r'(.)\1{2,}', r'\1', word)

    # remove users' tags
    if word.startswith('@') and len(word) > 1:
        word = '@user'
        lemma = word
    elif pos_t != '':
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos_t)
    else:
        lemma = word

    if word.startswith('#') and len(word) > 1:
        lemma = wordnet_lemmatizer.lemmatize(word[1:])

    word = word.replace(';', ' ')
    lemma = lemma.replace(';', ' ')
    return (word, lemma)


if __name__ == '__main__':
    tweet = ['I', 'love', 'to', 'read', 'books']
    tags = pos_tagging(tweet)
    tokens_lemmas = lemmatize(tags)
    print(tags)
    print(tokens_lemmas)

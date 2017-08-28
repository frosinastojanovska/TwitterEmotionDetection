from nltk import WordNetLemmatizer
from nltk import pos_tag
from nltk import map_tag


def pos_tagging(tweet):
    """
    Finds the pos tag with nltk post_tag function, and then maps them with
    a tag required for the lemmatize function.

    :param tweet: List of words (tokens) represented like strings
    :type tweet: list
    :return: List of tuple (word, tag)
    :rtype: list
    """
    dict_tags = {'ADJ': 'a', 'ADJ_SAT': 's', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
    tokens_tags = pos_tag(tweet)
    for (i, (token, tag)) in zip(range(len(tokens_tags)), tokens_tags):
        mapped = map_tag('en-ptb', 'universal', tag)
        if mapped in dict_tags:
            tokens_tags[i] = (token, dict_tags[mapped])
        else:
            tokens_tags[i] = (token, '')

    return tokens_tags


def lemmatize(tweet):
    """
    Finds lemma of words in tweets tagged with the required post tag that are mapped
    with the pos_tagging function: ['a', 's', 'r', 'n', 'v']

    :param tweet:
    :type tweet: list (string, string)
    :return: list of tuples (word, lemma)
    :rtype: list
    """
    tweet_list = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for token_tag in tweet:
        word = token_tag[0]
        pos_t = token_tag[1]
        if pos_t != '':
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos_t)
        else:
            lemma = word
        word = word.replace(";", " ")
        lemma = lemma.replace(";", " ")
        tweet_list.append((word, lemma))
    return tweet_list


if __name__ == '__main__':
    tweet = ['I', 'love', 'to', 'read', 'books']
    tags = pos_tagging(tweet)
    tokens_lemmas = lemmatize(tags)
    print(tags)
    print(tokens_lemmas)

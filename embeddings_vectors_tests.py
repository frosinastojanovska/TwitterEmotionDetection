import os
from gensim.models import KeyedVectors
from scipy.spatial import distance


def original_glove_embeddings():
    glove_file = 'data/glove.twitter.27B.200d.txt'
    word2vec_file = 'data/glove.twitter.27B.200d.txt.word2vec'
    if not os.path.exists(word2vec_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_file, word2vec_file)
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    # model.save_word2vec_format(word2vec_file + '.bin', binary=True)
    sim = emotion_arithmetic_test_closeness(model)
    print(sim)
    print(sum(sim)/len(sim))
    sim = emotion_arithmetic_test_opposite(model)
    print(sim)
    print(sum(sim) / len(sim))


def original_emoji2vec_embeddings():
    emoji2vec_file = 'data/emoji2vec.txt'
    model = KeyedVectors.load_word2vec_format(emoji2vec_file, binary=False)
    print(model.most_similar(positive=['💗'], negative=['😞'], topn=10))


def edim_word2vec_embeddings():
    original_file = 'data/w2v.twitter.edinburgh10M.400d.txt'
    word2vec_file = 'data/w2v.twitter.edinburgh10M.400d.txt.word2vec'
    if not os.path.exists(word2vec_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(original_file, word2vec_file)
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    # model.save_word2vec_format(word2vec_file + '.bin', binary=True)
    sim = emotion_arithmetic_test_closeness(model)
    print(sim)
    print(sum(sim) / len(sim))
    sim = emotion_arithmetic_test_opposite(model)
    print(sim)
    print(sum(sim) / len(sim))


def emotion_arithmetic_test_closeness(model):
    vectors = [('joy', 'trust', 'love'), ('joy', 'fear', 'guilt'), ('joy', 'surprise', 'delight'),
               ('trust', 'fear', 'submission'), ('trust', 'surprise', 'curiosity'),
               ('trust', 'sadness', 'sentimentality'),
               ('fear', 'surprise', 'awe'), ('fear', 'sadness', 'despair'), ('fear', 'disgust', 'shame'),
               ('surprise', 'sadness', 'disappointment'), ('surprise', 'disgust', 'unbelief'),
               ('surprise', 'anger', 'outage'),
               ('sadness', 'disgust', 'remorse'), ('sadness', 'anger', 'envy'), ('sadness', 'anticipation', 'pessimism'),
               ('disgust', 'anger', 'contempt'), ('disgust', 'anticipation', 'cynicism'),
               # ('disgust', 'joy', 'morbidness'),
               ('anger', 'anticipation', 'aggression'), ('anger', 'joy', 'pride'), ('anger', 'trust', 'dominance'),
               ('anticipation', 'joy', 'optimism'), ('anticipation', 'trust', 'hope'),
               ('anticipation', 'fear', 'anxiety')
               ]
    sim = []
    for x, y, z in vectors:
        _sim = 1 - distance.cosine(model[x] + model[y], model[z])
        sim.append(_sim)

    return sim


def emotion_arithmetic_test_opposite(model):
    vectors = [('anticipation', 'surprise'), ('anger', 'fear'),
               ('disgust', 'trust'), ('sadness', 'joy')]
    sim = []
    for x, y in vectors:
        _sim = 1 - distance.cosine(model[x], model[y])
        sim.append(_sim)

    return sim


if __name__ == '__main__':
    # original_glove_embeddings()
    # original_emoji2vec_embeddings()
    edim_word2vec_embeddings()

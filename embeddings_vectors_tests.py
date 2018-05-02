import os
from gensim.models import KeyedVectors


def original_glove_embeddings():
    glove_file = 'data/glove.twitter.27B.200d.txt'
    word2vec_file = 'data/glove.twitter.27B.200d.txt.word2vec'
    if not os.path.exists(word2vec_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_file, word2vec_file)
    model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    # model.save_word2vec_format(word2vec_file + '.bin', binary=True)
    # print(model.most_similar(positive=['happy'], negative=[], topn=10))
    # print(model.most_similar(positive=['sad'], negative=[], topn=10))
    # print(model.most_similar(positive=['angry'], negative=[], topn=10))
    # print(model.most_similar(positive=['joy'], negative=[], topn=10))
    print(model.most_similar(positive=['joy', 'trust'], negative=[], topn=10))
    # print(model.most_similar(positive=['joy', 'fear'], negative=[], topn=10))
    print(model.most_similar(positive=['anger', 'joy'], negative=[], topn=10))


def original_emoji2vec_embeddings():
    emoji2vec_file = 'data/emoji2vec.txt'
    model = KeyedVectors.load_word2vec_format(emoji2vec_file, binary=False)
    print(model.most_similar(positive=['💗'], negative=['😞'], topn=10))


if __name__ == '__main__':
    original_glove_embeddings()
    # original_emoji2vec_embeddings()

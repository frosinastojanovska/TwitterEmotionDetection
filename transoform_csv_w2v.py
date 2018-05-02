import numpy as np


def read_file():
    file = 'data/w2v.twitter.edinburgh10M.400d.csv'
    output_file = 'data/w2v.twitter.edinburgh10M.400d.txt'
    vocab_size = 258917
    embedding_dim = 400

    word2index = {}  # word to word-index
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    idx = 1  # first row set to zero (for unknown words)

    with open(file, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            data = line.split('\t')
            word = data[-1]
            coefs = np.asarray(data[:-1], dtype='float32')
            word2index[word] = idx
            embedding_matrix[idx] = np.asarray(coefs)
            idx += 1
            if idx % 1000 == 0:
                print(idx)
            line = doc.readline()

    print('Write word vectors to', output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, i in word2index.items():
            f.write(word)
            f.write(" ")
            f.write(" ".join(map(str, embedding_matrix[i])))
            f.write("\n")


if __name__ == '__main__':
    read_file()

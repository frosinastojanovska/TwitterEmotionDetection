import keras as k
import keras.layers as kl
from keras.layers.merge import concatenate


def create_merged_model(num_classes, input_shape1, input_shape2, embedding_matrix=None,
                        lexicon_matrix=None, max_length=None):
    """ Creates model of specified model type for classification of emotions with Glove and lexicon embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape1: shape of the first input
    :type input_shape1: tuple
    :param input_shape2: shape of the second input
    :type input_shape2: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param lexicon_matrix: embedding matrix from lexicon for Keras Embedding layer
    :type lexicon_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: deep learning model
    """
    model = cnn_model(num_classes, input_shape1, input_shape2, embedding_matrix, lexicon_matrix, max_length)
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.categorical_accuracy,
                           k.metrics.mae,
                           k.metrics.top_k_categorical_accuracy])
    return model


def cnn_model(num_classes, input_shape1, input_shape2, embedding_matrix, lexicon_matrix, max_length):
    """ Creates CNN model for classification of emotions with Glove and lexicon embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :type input_shape1: tuple
    :param input_shape2: shape of the second input
    :type input_shape2: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param lexicon_matrix: embedding matrix from lexicon for Keras Embedding layer
    :type lexicon_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: CNN model
    """
    first_model = k.Input(shape=input_shape1)
    embedding = kl.Embedding(input_dim=embedding_matrix.shape[0],
                             output_dim=embedding_matrix.shape[1],
                             weights=[embedding_matrix],
                             input_length=max_length,
                             trainable=False,
                             name='embedding_layer')(first_model)
    conv1 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape1)(embedding)
    pool = kl.MaxPooling1D()(conv1)
    conv2 = kl.Convolution1D(64, 3, activation='relu')(pool)
    pool2 = kl.GlobalMaxPooling1D()(conv2)

    second_model = k.Input(shape=input_shape2)
    embedding2 = kl.Embedding(input_dim=lexicon_matrix.shape[0],
                              output_dim=lexicon_matrix.shape[1],
                              weights=[lexicon_matrix],
                              input_length=max_length,
                              trainable=False,
                              name='lexicon_embedding_layer')(second_model)
    conv21 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape2)(embedding2)
    pool21 = kl.MaxPooling1D()(conv21)
    conv22 = kl.Convolution1D(64, 3, activation='relu')(pool21)
    pool22 = kl.GlobalMaxPooling1D()(conv22)

    merge = concatenate([pool2, pool22])
    dense = kl.Dense(128)(merge)
    drop_out = kl.Dropout(0.2)(dense)
    activation = kl.Activation('relu')(drop_out)
    output = kl.Dense(num_classes, activation='sigmoid')(activation)

    model = k.Model(inputs=[first_model, second_model], outputs=output)

    return model

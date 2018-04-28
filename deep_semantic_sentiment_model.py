import keras as k
import keras.layers as kl
from keras.layers.merge import concatenate


def create_merged_model(model_type, num_classes, input_shape1, input_shape2, embedding_matrix=None,
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
    if model_type == 'cnn':
        model = cnn_model(num_classes, input_shape1, input_shape2, embedding_matrix, lexicon_matrix, max_length)
    elif model_type == 'cnn_bi_lstm':
        model = cnn_bi_lstm_model(num_classes, input_shape1, input_shape2, embedding_matrix, lexicon_matrix, max_length)
    elif model_type == 'lexicon_cnn_bi_lstm':
        model = lexicon_cnn_bi_lstm_model(num_classes, input_shape1, embedding_matrix, max_length)
        model.load_weights('models/embedding_sentiment_model.h5', by_name=True)
    else:
        raise ValueError('Model type should be one of the following: cnn, or cnn_bi_lstm')
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])
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
    pool11 = kl.MaxPooling1D()(conv1)

    second_model = k.Input(shape=input_shape2)
    embedding2 = kl.Embedding(input_dim=lexicon_matrix.shape[0],
                              output_dim=lexicon_matrix.shape[1],
                              weights=[lexicon_matrix],
                              input_length=max_length,
                              trainable=False,
                              name='lexicon_embedding_layer')(second_model)
    conv21 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape2)(embedding2)
    pool21 = kl.MaxPooling1D()(conv21)

    merge = concatenate([pool11, pool21])
    conv2 = kl.Convolution1D(64, 3, activation='relu')(merge)
    pool_merged = kl.GlobalMaxPooling1D()(conv2)
    dense = kl.Dense(128)(pool_merged)
    drop_out = kl.Dropout(0.2)(dense)
    activation = kl.Activation('relu')(drop_out)
    output = kl.Dense(num_classes, activation='sigmoid')(activation)

    model = k.Model(inputs=[first_model, second_model], outputs=output)

    return model


def cnn_bi_lstm_model(num_classes, input_shape1, input_shape2, embedding_matrix, lexicon_matrix, max_length):
    """ Creates CNN + Bidirectional LSTM model for classification of emotions with Glove and lexicon embeddings

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
    :return: CNN+Bidirectional LSTM model
    """
    first_model = k.Input(shape=input_shape1)
    embedding = kl.Embedding(input_dim=embedding_matrix.shape[0],
                             output_dim=embedding_matrix.shape[1],
                             weights=[embedding_matrix],
                             input_length=max_length,
                             trainable=False,
                             name='embedding_layer')(first_model)
    conv11 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape1)(embedding)
    pool11 = kl.MaxPooling1D()(conv11)
    conv12 = kl.Convolution1D(64, 3, activation='relu')(pool11)
    pool12 = kl.MaxPooling1D()(conv12)
    drop_out1 = kl.Dropout(0.2)(pool12)
    bi1 = kl.Bidirectional(kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(drop_out1)

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
    pool22 = kl.MaxPooling1D()(conv22)
    drop_out2 = kl.Dropout(0.2)(pool22)
    bi2 = kl.Bidirectional(kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(drop_out2)

    merge = concatenate([bi1, bi2])
    output = kl.Dense(num_classes, activation='sigmoid')(merge)

    model = k.Model(inputs=[first_model, second_model], outputs=output)

    return model


def embedding_to_sentiment(input_shape, num_lexicon):
    input_layer = kl.Input(shape=input_shape)
    embedding = kl.Dense(input_shape[0], name='sem2sent_input_layer')(input_layer)
    activation0 = kl.LeakyReLU(name='sem2sent_activation0')(embedding)
    layer1 = kl.Dense(128, name='sem2sent_layer1')(activation0)
    activation1 = kl.LeakyReLU(name='sem2sent_activation1')(layer1)
    layer2 = kl.Dense(64, name='sem2sent_layer2')(activation1)
    activation2 = kl.LeakyReLU(name='sem2sent_activation2')(layer2)
    layer3 = kl.Dense(num_lexicon, name='sem2sent_layer3')(activation2)
    activation3 = kl.LeakyReLU(name='sem2sent_activation3')(layer3)
    return input_layer, activation3


def embedding_to_sentiment_model(input_shape, num_lexicon):
    input_layer, embed_to_sent = embedding_to_sentiment(input_shape, num_lexicon)
    model = k.Model(inputs=input_layer, outputs=embed_to_sent)
    return model


def lexicon_cnn_bi_lstm_model(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates CNN + Bidirectional LSTM model for classification of emotions with Glove and learned lexicon embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: CNN+Bidirectional LSTM model
    """
    input_layer = k.Input(shape=input_shape)
    embedding = kl.Embedding(input_dim=embedding_matrix.shape[0],
                             output_dim=embedding_matrix.shape[1],
                             weights=[embedding_matrix],
                             input_length=max_length,
                             trainable=False,
                             name='embedding_layer')(input_layer)
    conv11 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape)(embedding)
    pool11 = kl.MaxPooling1D()(conv11)
    conv12 = kl.Convolution1D(64, 3, activation='relu')(pool11)
    pool12 = kl.MaxPooling1D()(conv12)
    drop_out1 = kl.Dropout(0.2)(pool12)
    bi1 = kl.Bidirectional(kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(drop_out1)

    layer0 = kl.TimeDistributed(kl.Dense(input_shape[0], name='sem2sent_input_layer'))(embedding)
    activation0 = kl.TimeDistributed(kl.PReLU(name='sem2sent_activation0'))(layer0)
    layer1 = kl.TimeDistributed(kl.Dense(128, name='sem2sent_layer1'))(activation0)
    activation1 = kl.TimeDistributed(kl.PReLU(name='sem2sent_activation1'))(layer1)
    layer2 = kl.TimeDistributed(kl.Dense(64, name='sem2sent_layer2'))(activation1)
    activation2 = kl.TimeDistributed(kl.PReLU(name='sem2sent_activation2'))(layer2)
    layer3 = kl.TimeDistributed(kl.Dense(3, name='sem2sent_layer3'))(activation2)
    activation3 = kl.TimeDistributed(kl.PReLU(name='sem2sent_activation3'))(layer3)

    conv21 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape)(activation3)
    pool21 = kl.MaxPooling1D()(conv21)
    conv22 = kl.Convolution1D(64, 3, activation='relu')(pool21)
    pool22 = kl.MaxPooling1D()(conv22)
    drop_out2 = kl.Dropout(0.2)(activation3)
    bi2 = kl.Bidirectional(kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(drop_out2)

    merge = concatenate([bi1, bi2])
    output = kl.Dense(num_classes, activation='sigmoid')(merge)

    model = k.Model(inputs=input_layer, outputs=output)

    return model


def top_3_accuracy(y_true, y_pred):
    return k.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

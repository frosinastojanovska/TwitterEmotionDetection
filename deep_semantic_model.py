import keras as k
import keras.layers as kl


def create_model(model_type, num_classes, input_shape, embedding_matrix=None, max_length=None):
    """ Creates model of specified model type for classification of emotions with Glove embeddings

    :param model_type: type of deep learning model to be instantiated
    :type model_type: str
    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: deep learning model
    """
    if model_type == 'cnn':
        model = cnn_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'lstm1':
        model = lstm_model_1(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'lstm2':
        model = lstm_model_2(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'bi_lstm':
        model = bidirectional_lstm_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'gru':
        model = gru_model(num_classes, input_shape, embedding_matrix, max_length)
    else:
        raise ValueError('Model type should be one of the following: cnn, lstm1, lstm2, bi_lstm or gru')
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.categorical_accuracy,
                           k.metrics.mae,
                           k.metrics.top_k_categorical_accuracy])
    return model


def cnn_model(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates CNN model for classification of emotions with Glove embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: CNN model
    """
    model = k.Sequential()

    model.add(kl.Embedding(input_dim=embedding_matrix.shape[0],
                           output_dim=embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=max_length,
                           trainable=False,
                           name='embedding_layer'))

    model.add(kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.GlobalMaxPooling1D())
    model.add(kl.Dense(128))
    model.add(kl.Dropout(0.2))
    model.add(kl.Activation('relu'))

    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def lstm_model_1(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates LSTM model for classification of emotions with Glove embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: LSTM model
    """
    model = k.Sequential()

    model.add(kl.Embedding(input_dim=embedding_matrix.shape[0],
                           output_dim=embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=max_length,
                           trainable=False,
                           name='embedding_layer'))

    model.add(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def lstm_model_2(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates LSTM model for classification of emotions with Glove embeddings with additional hidden layer

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: LSTM model
    """
    model = k.Sequential()

    model.add(kl.Embedding(input_dim=embedding_matrix.shape[0],
                           output_dim=embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=max_length,
                           trainable=False,
                           name='embedding_layer'))

    model.add(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(kl.Dense(128))
    model.add(kl.Dropout(0.2))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def bidirectional_lstm_model(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates Bidirectional LSTM model for classification of emotions with Glove embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: Bidirectional LSTM model
    """
    model = k.Sequential()

    model.add(kl.Embedding(input_dim=embedding_matrix.shape[0],
                           output_dim=embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=max_length,
                           trainable=False,
                           name='embedding_layer'))

    model.add(kl.Bidirectional(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def gru_model(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates GRU model for classification of emotions with Glove embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :param embedding_matrix: embedding matrix for Keras Embedding layer
    :type embedding_matrix: numpy.array
    :param max_length: maximum length of the text sequence
    :type max_length: int
    :return: GRU model
    """
    model = k.Sequential()

    model.add(kl.Embedding(input_dim=embedding_matrix.shape[0],
                           output_dim=embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=max_length,
                           trainable=False,
                           name='embedding_layer'))

    model.add(kl.GRU(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model

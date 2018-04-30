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
        model = lstm_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'lstm2':
        model = cnn_lstm_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'bi_lstm':
        model = cnn_bidirectional_lstm_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'gru':
        model = cnn_gru_model(num_classes, input_shape, embedding_matrix, max_length)
    elif model_type == 'attention_lstm':
        model = cnn_attention_lstm(num_classes, input_shape, embedding_matrix, max_length)
    else:
        raise ValueError('Model type should be one of the following: cnn, lstm1, lstm2, bi_lstm or gru')
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])
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


def lstm_model(num_classes, input_shape, embedding_matrix, max_length):
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
    model.add(kl.SpatialDropout1D(0.6))
    model.add(kl.LSTM(32, dropout=0.1, recurrent_dropout=0.2))
    model.add(kl.Dense(128))
    model.add(kl.Dropout(0.2))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def cnn_lstm_model(num_classes, input_shape, embedding_matrix, max_length):
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

    model.add(kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.MaxPooling1D())
    model.add(kl.Dropout(0.2))
    model.add(kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(kl.Dense(num_classes, activation='sigmoid'))

    return model


def cnn_bidirectional_lstm_model(num_classes, input_shape, embedding_matrix, max_length):
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
    model.add(kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling1D())
    model.add(kl.Dropout(0.1))
    model.add(kl.Bidirectional(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(kl.Dense(num_classes, activation='sigmoid'))

    return model


def cnn_gru_model(num_classes, input_shape, embedding_matrix, max_length):
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

    model.add(kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.MaxPooling1D())
    model.add(kl.Dropout(0.2))
    model.add(kl.GRU(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(kl.Dense(num_classes, activation='sigmoid'))

    return model


def cnn_attention_lstm(num_classes, input_shape, embedding_matrix, max_length):
    """ Creates LSTM model with attention layer for classification of emotions with Glove embeddings

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
    inputs = kl.Input(shape=input_shape)
    embeddings = kl.Embedding(input_dim=embedding_matrix.shape[0],
                              output_dim=embedding_matrix.shape[1],
                              weights=[embedding_matrix],
                              input_length=max_length,
                              trainable=False,
                              name='embedding_layer')(inputs)
    conv1 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape)(embeddings)
    pool1 = kl.MaxPooling1D()(conv1)
    conv2 = kl.Convolution1D(64, 3, activation='relu')(pool1)
    pool2 = kl.MaxPooling1D()(conv2)
    drop = kl.Dropout(0.2)(pool2)
    attention_mul = attention_3d_block(drop, int(drop.shape[1]), True)
    attention_mul = kl.LSTM(32, dropout=0.2, recurrent_dropout=0.2)(attention_mul)
    output = kl.Dense(num_classes, activation='sigmoid')(attention_mul)
    model = k.Model(input=[inputs], output=output)

    return model


# code taken from https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
def attention_3d_block(inputs, time_steps, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = kl.Permute((2, 1))(inputs)
    a = kl.Reshape((input_dim, time_steps))(a)  # this line is not useful. It's just to know which dimension is what.
    a = kl.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = kl.Lambda(lambda x: k.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = kl.RepeatVector(input_dim)(a)
    a_probs = kl.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = kl.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def top_3_accuracy(y_true, y_pred):
    return k.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

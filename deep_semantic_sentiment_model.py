import keras as k
import keras.layers as kl
from keras.regularizers import l2
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
                             trainable=True,
                             name='embedding_layer')(first_model)
    norm1 = kl.BatchNormalization()(embedding)
    conv1 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape1)(norm1)
    norm2 = kl.BatchNormalization()(conv1)
    pool11 = kl.MaxPooling1D()(norm2)
    conv12 = kl.Convolution1D(64, 3, activation='relu')(pool11)
    norm3 = kl.BatchNormalization()(conv12)
    pool12 = kl.GlobalMaxPooling1D()(norm3)

    second_model = k.Input(shape=input_shape2)
    embedding2 = kl.Embedding(input_dim=lexicon_matrix.shape[0],
                              output_dim=lexicon_matrix.shape[1],
                              weights=[lexicon_matrix],
                              input_length=max_length,
                              trainable=True,
                              name='lexicon_embedding_layer')(second_model)
    norm4 = kl.BatchNormalization()(embedding2)
    conv21 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape2)(norm4)
    norm5 = kl.BatchNormalization()(conv21)
    pool21 = kl.MaxPooling1D()(norm5)
    conv22 = kl.Convolution1D(64, 3, activation='relu')(pool21)
    norm6 = kl.BatchNormalization()(conv22)
    pool22 = kl.GlobalMaxPooling1D()(norm6)

    merged = concatenate([pool12, pool22])
    drop_out = kl.Dropout(0.2)(merged)
    dense = kl.Dense(128)(drop_out)
    drop_out2 = kl.Dropout(0.1)(dense)
    activation = kl.Activation('relu')(drop_out2)
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
    norm11 = kl.BatchNormalization()(conv11)
    pool11 = kl.MaxPooling1D()(norm11)
    conv12 = kl.Convolution1D(64, 3, activation='relu')(pool11)
    norm12 = kl.BatchNormalization()(conv12)
    pool12 = kl.MaxPooling1D()(norm12)
    drop_out1 = kl.Dropout(0.1)(pool12)
    bi1 = kl.Bidirectional(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(drop_out1)

    second_model = k.Input(shape=input_shape2)
    embedding2 = kl.Embedding(input_dim=lexicon_matrix.shape[0],
                              output_dim=lexicon_matrix.shape[1],
                              weights=[lexicon_matrix],
                              input_length=max_length,
                              trainable=False,
                              name='lexicon_embedding_layer')(second_model)
    norm21 = kl.BatchNormalization()(embedding2)
    flatten = kl.Flatten()(norm21)

    merge = concatenate([bi1, flatten], axis=1)
    output = kl.Dense(num_classes, activation='sigmoid')(merge)

    model = k.Model(inputs=[first_model, second_model], outputs=output)

    return model


def embedding_to_sentiment(input_shape, num_output):
    input_layer = kl.Input(shape=input_shape)
    embedding = kl.Dense(input_shape[0], name='sem2sent_input_layer')(input_layer)
    activation0 = kl.LeakyReLU(name='sem2sent_activation0')(embedding)
    layer1 = kl.Dense(128, name='sem2sent_layer1')(activation0)
    activation1 = kl.LeakyReLU(name='sem2sent_activation1')(layer1)
    drop1 = kl.Dropout(0.3, name='sem2sent_dropout1')(activation1)
    layer2 = kl.Dense(64, name='sem2sent_layer2')(drop1)
    activation2 = kl.LeakyReLU(name='sem2sent_activation2')(layer2)
    drop2 = kl.Dropout(0.3, name='sem2sent_dropout2')(activation2)
    layer3 = kl.Dense(num_output, name='sem2sent_layer3')(drop2)
    activation3 = kl.LeakyReLU(name='sem2sent_activation3')(layer3)
    return input_layer, activation3


def embedding_to_sentiment_model(input_shape, num_output):
    input_layer, embed_to_sent = embedding_to_sentiment(input_shape, num_output)
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
    norm1 = kl.BatchNormalization()(embedding)
    conv11 = kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape)(norm1)
    pool11 = kl.MaxPooling1D()(conv11)
    conv12 = kl.Convolution1D(64, 3, activation='relu')(pool11)
    pool12 = kl.MaxPooling1D()(conv12)

    layer0 = kl.TimeDistributed(kl.Dense(input_shape[0], name='sem2sent_input_layer', trainable=False))(embedding)
    activation0 = kl.TimeDistributed(kl.LeakyReLU(name='sem2sent_activation0', trainable=False))(layer0)
    layer1 = kl.TimeDistributed(kl.Dense(128, name='sem2sent_layer1', trainable=False))(activation0)
    activation1 = kl.TimeDistributed(kl.LeakyReLU(name='sem2sent_activation1', trainable=False))(layer1)
    drop1 = kl.TimeDistributed(kl.Dropout(0.3, name='sem2sent_dropout1', trainable=False))(activation1)
    layer2 = kl.TimeDistributed(kl.Dense(64, name='sem2sent_layer2', trainable=False))(drop1)
    activation2 = kl.TimeDistributed(kl.LeakyReLU(name='sem2sent_activation2', trainable=False))(layer2)
    drop2 = kl.TimeDistributed(kl.Dropout(0.3, name='sem2sent_dropout2', trainable=False))(activation2)
    layer3 = kl.TimeDistributed(kl.Dense(3, name='sem2sent_layer3', trainable=False))(drop2)
    activation3 = kl.TimeDistributed(kl.LeakyReLU(name='sem2sent_activation3', trainable=False))(layer3)

    norm2 = kl.BatchNormalization()(activation3)
    conv21 = kl.Convolution1D(32, 3, activation='relu')(norm2)
    pool21 = kl.MaxPooling1D()(conv21)
    conv22 = kl.Convolution1D(64, 3, activation='relu')(pool21)
    pool22 = kl.MaxPooling1D()(conv22)

    merge = concatenate([pool12, pool22])
    # drop_out2 = kl.Dropout(0.5)(merge)
    bi = kl.Bidirectional(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                  recurrent_regularizer=l2()))(merge)
    # output1 = kl.Dense(128, activation='relu')(bi)
    # drop_output = kl.Dropout(0.2)(output1)
    output2 = kl.Dense(num_classes, activation='sigmoid')(bi)

    model = k.Model(inputs=input_layer, outputs=output2)

    return model


def top_3_accuracy(y_true, y_pred):
    return k.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

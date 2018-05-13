import pandas as pd
import numpy as np
from visualize_confusion_matrix import plot_confusion_matrix


def load_data(file_name, col):
    df = pd.read_table(file_name, usecols=[col + 2])
    return df


def create_confusion_matrix(df_g, df_p):
    matrix = np.zeros((2, 2))
    for i in range(df_g.shape[0]):
        a = df_g.at[i, df_g.columns[0]]
        b = df_p.at[i, df_p.columns[0]]
        matrix[a, b] = matrix[a, b] + 1
        '''
        if a == 1 and b == 1:
            matrix[0, 0] = matrix[0, 0] + 1
        elif a == 0 and b == 0:
            matrix[1, 1] = matrix[1, 1] + 1
        elif a == 1 and b == 0:
            matrix[0, 1] = matrix[0, 1] + 1
        else:
            matrix[1, 0] = matrix[1, 0] + 1
        '''
    '''
    for i in range(df_g.shape[0]):
        for j in range(df_g.shape[1]):
            for k in range(df_g.shape[1]):
                if df_g.at[i, j] == 1 and df_p.at[i, k] == 1:
                    matrix[j, k] = matrix[j, k] + 1
    '''
    return np.array(matrix, dtype=np.int)


ground_truth_file = 'data/2018-E-c-En-test-gold.txt'
predictions_file = 'data/predictions_binary.txt'
confusion_matrix_anger = create_confusion_matrix(load_data(ground_truth_file, 0), load_data(predictions_file, 0))
confusion_matrix_anticipation = create_confusion_matrix(load_data(ground_truth_file, 1), load_data(predictions_file, 1))
confusion_matrix_disgust = create_confusion_matrix(load_data(ground_truth_file, 2), load_data(predictions_file, 2))
confusion_matrix_fear = create_confusion_matrix(load_data(ground_truth_file, 3), load_data(predictions_file, 3))
confusion_matrix_joy = create_confusion_matrix(load_data(ground_truth_file, 4), load_data(predictions_file, 4))
confusion_matrix_love = create_confusion_matrix(load_data(ground_truth_file, 5), load_data(predictions_file, 5))
confusion_matrix_optimism = create_confusion_matrix(load_data(ground_truth_file, 6), load_data(predictions_file, 6))
confusion_matrix_pessimism = create_confusion_matrix(load_data(ground_truth_file, 7), load_data(predictions_file, 7))
confusion_matrix_sadness = create_confusion_matrix(load_data(ground_truth_file, 8), load_data(predictions_file, 8))
confusion_matrix_surprise = create_confusion_matrix(load_data(ground_truth_file, 9), load_data(predictions_file, 9))
confusion_matrix_trust = create_confusion_matrix(load_data(ground_truth_file, 10), load_data(predictions_file, 10))
plot_confusion_matrix([confusion_matrix_anger, confusion_matrix_anticipation, confusion_matrix_disgust,
                       confusion_matrix_fear, confusion_matrix_joy, confusion_matrix_love, confusion_matrix_optimism,
                       confusion_matrix_pessimism, confusion_matrix_sadness, confusion_matrix_surprise,
                       confusion_matrix_trust],
                      ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pesimism', 'sadness',
                       'surprise', 'trust'],
                      ['No', 'Yes'])

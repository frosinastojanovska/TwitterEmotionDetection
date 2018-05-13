import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(matrices, titles, classes):
    font = {'fontname': 'Cambria', 'size': 28}
    for matrix, title in zip(matrices, titles):
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title, weight='bold', **font)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, **font)
        plt.yticks(tick_marks, classes, **font)

        fmt = 'd'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black", **font)

        plt.tight_layout()
        plt.ylabel('True label', **font)
        plt.xlabel('Predicted label', **font)
        plt.show()


if __name__ == '__main__':
    """
       first row -> neutral
       second row -> anger
       third row -> fear
       forth row -> sadness
       fifth row -> joy
    """
    classes = ['neutral', 'anger', 'fear', 'sadness', 'joy']
    confusion_matrix_random_forest = np.array([[109, 3, 20, 1, 5],
                                               [4, 1061, 134, 85, 44],
                                               [6, 78, 1381, 121, 52],
                                               [5, 82, 180, 802, 36],
                                               [5, 33, 82, 14, 1005]])
    confusion_matrix_svm = np.array([[108, 5, 24, 0, 1],
                                     [6, 1053, 193, 52, 24],
                                     [2, 50, 1456, 97, 33],
                                     [6, 59, 177, 846, 17],
                                     [5, 24, 146, 12, 952]])
    confusion_matrix_lda = np.array([[98, 0, 39, 1, 0],
                                     [0, 1037, 233, 50, 8],
                                     [0, 22, 1500, 105, 11],
                                     [0, 43, 184, 869, 9],
                                     [0, 7, 178, 5, 949]])
    confusion_matrix_perceptron = np.array([[0, 31, 51, 33, 23],
                                            [0, 1041, 153, 84, 50],
                                            [0, 90, 1366, 120, 162],
                                            [0, 67, 151, 849, 38],
                                            [0, 50, 95, 19, 975]])

    plot_confusion_matrix([confusion_matrix_random_forest,
                          confusion_matrix_lda,
                          confusion_matrix_svm,
                          confusion_matrix_perceptron],
                          ['Random Forest', 'SVM', 'LDA', 'Multilayer Perceptron'],
                          classes)

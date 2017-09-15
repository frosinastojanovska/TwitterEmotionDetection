import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(matrix1, matrix2, matrix3, matrix4, classes):
    font = {'fontname': 'Calibri', 'size': 20}
    for matrix, title in zip([matrix1, matrix2, matrix3, matrix4], ['Random Forest', 'SVM', 'LDA', 'Multilayer Perceptron']):
        plt.figure()
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
    confusion_matrix_random_forest = np.array([[86, 13, 18, 4, 17],
                                               [33, 1051, 101, 94, 49],
                                               [40, 81, 1309, 142, 66],
                                               [18, 78, 162, 786, 61],
                                               [38, 53, 75, 37, 936]])
    confusion_matrix_svm = np.array([[35, 46, 42, 1, 14],
                                     [6, 1092, 137, 53, 40],
                                     [3, 129, 1362, 104, 40],
                                     [3, 106, 154, 798, 44],
                                     [8, 95, 116, 20, 900]])
    confusion_matrix_lda = np.array([[41, 4, 82, 1, 10],
                                     [14, 1020, 201, 58, 35],
                                     [6, 39, 1452, 104, 37],
                                     [7, 47, 187, 826, 38],
                                     [10, 28, 181, 18, 902]])
    confusion_matrix_perceptron = np.array([[0, 49, 31, 26, 32],
                                            [0, 1003, 119, 109, 97],
                                            [0, 118, 1263, 153, 104],
                                            [0, 128, 155, 740, 82],
                                            [0, 83, 94, 68, 894]])

    plot_confusion_matrix(confusion_matrix_random_forest,
                          confusion_matrix_lda,
                          confusion_matrix_svm,
                          confusion_matrix_perceptron, classes)

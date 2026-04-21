import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        bin_loss=0
        n=len(y_true)
        for i in range(n):
            y_pred[i]+=10**(-7)
            bin_loss += (y_true[i]*math.log(y_pred[i]) + (1-y_true[i])*math.log((1-y_pred[i])))
        return round((-1/n)*bin_loss,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        cross_loss=0
        n=len(y_true)
        c=len(y_true[0])
        for i in range(n):
            for j in range(c):
                y_pred[i][j]+=10**(-7)
                cross_loss+=y_true[i][j]*math.log(y_pred[i][j])

            
        return round((-1/n)*cross_loss,4)

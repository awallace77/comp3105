import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigh
from cvxopt import matrix, solvers
from A3helpers import *
import pandas as pd

'''
    COMP3105 Assignment 3
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 101260693
    Due: November 16th, 2025
'''

### QUESTION 1
def minMulDev(X, Y):
    """
        Computes the optimal weights using multinomial deviance loss
        Args:
            X: n-by-d input matrix
            Y: n-by-k label matrix where each row is a one-hot encoding of the label
        Returns:
            d-by-k matrix of weights corresponding to the solution of the multinomial deviance loss  
    """

    n, d = X.shape
    _, k = Y.shape

    # Define objective
    def obj_func(u):

        W = u.reshape(d, k)

        weights_inputs = (X @ W).T 
        labels_weights_inputs = (Y @ W.T @ X.T)

        first_term = np.sum(logsumexp(weights_inputs, axis=0))
        second_term = np.trace(labels_weights_inputs)

        loss = (first_term - second_term) / n
        return loss
    

    # Find solution 
    u = np.zeros(d * k)
    sol = minimize(obj_func, u)  
    W = sol['x'].reshape(d,k)  # make it dxk

    return W


def classify(Xtest, W):
    """
        Calculates the predicted class for a given test dataset and model.
        Args:
            Xtest: m-by-d input matrix
            W: d-by-k matrix of weights
        Returns:
            m-by-k predication matrix
    """
    Z = Xtest @ W
    max_values = np.max(Z, axis=1, keepdims=True)
    Yhat = (Z == max_values).astype(int)

    return Yhat

def calculateAcc(Yhat, Y):
    """
        Calculates the accuracy of a prediction
        Args:
            Yhat: m-by-k prediction matrix
            Y: m-by-k label matrix
        Returns:
            Scalar accuracy of the prediction
    """
    m, k = Y.shape
    num_correct = np.sum((Yhat == Y).all(axis=1))
    acc = num_correct / m

    return acc


### QUESTION 2
def PCA(X, k):
    """
        Calculates the top k projecting vectors for dimensionality reduction
        Args:
            X: n-by-d input matrix
            k: scalar integer 1 <= k <= d
        Returns:
            k-by-d matrix of rows corresponding to the top-k projecting directions with the largest variances.
    """
    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(d, 1)
    X = (X - (mean.T * np.ones(X.shape)))
    eigen_values, eigen_vectors = eigh(X.T @ X, subset_by_index=[d - k, d - 1])
    U = np.array((eigen_vectors).T)

    return U

if __name__ == "__main__":

    # Small test matrices
    X = np.array([
        [1, 2],
        [3, 4]
    ])

    Y = np.array([
        [1, 0],
        [0, 1]
    ])

    W = np.array([
        [1, 2],
        [1, 1]
    ])

    Yhat = np.array([
        [0, 1],
        [0, 1]
    ])

    W = minMulDev(X, Y)
    classify(X, W)
    print(calculateAcc(Yhat, Y))

    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    # U = PCA(X, 3)
    # print(U)


    data_matrix = np.loadtxt('./a3/src/A3train.csv', delimiter=',')
    n, d = data_matrix.shape
    U = PCA(data_matrix, 3)
    plotImgs(U)
    


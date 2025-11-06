import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from cvxopt import matrix, solvers
from A3helpers import *
import pandas as pd

'''
    COMP3105 Assignment 3
    Group 51:
        Andrew Wallace - 101210291
    Due: November 16th, 2025
'''

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

    # [ W.T x_1  W.T x_2 ... ]
    # n = 2
    # # WtX = (W.T @ X.T)
    # WtX = (X @ W).T 
    # YWtXt = (Y @ W.T @ X.T)
    # print(WtX)
    # print(YWtXt)
    # first_term = np.sum(logsumexp(WtX, axis=0))
    # second_term = np.trace(YWtXt)
    # loss = (first_term - second_term) / n

    # print(f"First Term: {first_term}")
    # print(f"Second Term: {second_term}")
    # print(f"Res: {loss}")


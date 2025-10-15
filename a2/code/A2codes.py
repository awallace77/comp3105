import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import *

'''
    COMP3105
    Assignment 2
    Andrew Wallace - 101210291
    Christer Henrysson - 101260693
    Due: October 19th, 2025
'''

# Question 1: Binary Classifier (Primal Form)
# (a) 
def minExpLinear(X, y, lamb):
    """
        This function computes the best model with ExpLinear loss and L2 regularization.
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
        Returns:
            A tuple of optimal model parameters (w, w0) where w is d-by-1 and w0 is a scalar
    """

    d = X.shape[1]

    # Define objective
    def obj_func(u):

        w0 = u[-1]  # last unknown is w0
        w = u[:-1]  # the first d dimensions are w
        w = w[:, None]  # make it d-by-1

        m = y * (X @ w + w0) 
        loss = np.sum(np.where(m <= 0, 1 - m, np.exp(-m)))
        reg = 0.5 * lamb * float((w.T @ w)[0][0])
        
        return loss + reg

    # Initial guess of unknowns, shouldn't matter for convex problem
    u0 = np.ones(d + 1)  

    sol = minimize(obj_func, u0)  # objective function + initial guess as inputs

    # Get the solution
    w = sol['x'][:-1][:, None]  # make it d-by-1
    w0 = sol['x'][-1]

    return w, w0


# Question 1(b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    """
        This function computes the best model with hinge loss and L2 regularization.
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
            stabilizer: small positive stabilizer added to diagonal of P
        Returns:
            A tuple of optimal model parameters (w, w0) where w is d-by-1 and w0 is a scalar
    """
    n, d = X.shape

    # construct P, q, G, and h for qp 
    q = np.concatenate([np.zeros(d + 1), np.ones(n)])
    G1 = np.hstack([np.zeros((n, d)), np.zeros((n, 1)), -np.eye(n)])
    G2 = np.hstack([-np.diag(y.reshape(n,)) @ X, -y, -np.eye(n)])
    G = np.vstack([G1, G2])
    h = np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))])

    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = lamb * np.eye(d)  # only w is regularized
    P = P + stabilizer * np.eye(n + d + 1)  # add stabilizer

    # Convert to cvx matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u = np.array(sol['x']).flatten()
    w = u[:d][:, None] # make it dx1
    w0 = u[d] 

    return  w, w0


# Question 1 (c)
def classify(Xtest, w, w0):
    """
        This function computes the prediction of a given model.
        Args:
            X: mxd test matrix
            w: dx1 model parameters
            w0: scalar bias for model
        Returns:
            A mx1 vector of predictions for the given model 
    """
    y_hat = np.sign(Xtest @ w + w0)
    return y_hat

# Question 1(d)
def synExperimentsRegularize():
    """
        This function runs a synthetic experiment for the expLinear and hinge losses with L2 regularization.
        Returns:
            A 4x6 matrix of average training accuracies and a 4x6 matrix of average test accuracies 
    """
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    np.random.seed(51)
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minExpLinear(Xtrain, ytrain, lamb)
                train_acc_explinear[i, j, r] = np.mean(ytrain == classify(Xtrain, w, w0))
                test_acc_explinear[i, j, r] = np.mean(ytest == classify(Xtest, w, w0)) 

                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(ytrain == classify(Xtrain, w, w0))
                test_acc_hinge[i, j, r] = np.mean(ytest == classify(Xtest, w, w0))

    # Calculate avg training acc for explinear & hinge loss
    avg_train_acc_explinear = np.mean(train_acc_explinear, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_train_acc = np.hstack([avg_train_acc_explinear, avg_train_acc_hinge])
    
    # Calculate avg test acc for explinear & hinge loss
    avg_test_acc_explinear = np.mean(test_acc_explinear, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)
    avg_test_acc = np.hstack([avg_test_acc_explinear, avg_test_acc_hinge])

    return avg_test_acc, avg_train_acc

if __name__ == "__main__":

    # Question 1 (d)
    synExperimentsRegularize()



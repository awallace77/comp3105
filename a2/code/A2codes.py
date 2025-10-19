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


# Question 1 (b)
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

    return avg_train_acc, avg_test_acc


# Question 2: Binary Classification (Adjoint Form)
# Question 2 (a)
def adjExpLinear(X, y, lamb, kernel_func):
    """
        This function computes the solution of the adjoint form of the regularized ExpLinear loss
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
            kernel_func: a callable kernel function
        Returns:
            A tuple of optimal model parameters (a, a0) where a is n-by-1 and a0 is a scalar
    """
    n, _ = X.shape

    # Define objective
    def obj_func(u):

        a0 = u[-1]  # last unknown is a0
        a = u[:-1]  # the first n dimensions are a
        a = a[:, None]  # make it n-by-1
        
        K = kernel_func(X, X) # kernel matrix
        m = y * (K.T @ a + a0) # margin

        mask = (m <= 0) # our condition
        result = np.empty_like(m)
        result[mask] = 1 - m[mask] # values <=0 are set to 1-m
        result[~mask] = np.exp(-m[~mask]) # values > 0 are set to e^{-m}

        loss = np.sum(result) # compute the loss
        reg = 0.5 * lamb * float((a.T @ K @ a)[0][0])
        
        return loss + reg

    # Initial guess of unknowns
    u0 = np.ones(n + 1)  

    sol = minimize(obj_func, u0)  # objective function + initial guess as inputs

    # Get the solution
    a = sol['x'][:-1][:, None]  # make it n-by-1
    a0 = sol['x'][-1]

    return a, a0

# Question 2 (b)
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """
        This function computes the solution of the adjoint form of the regularized Hinge loss
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
            kernel_func: a callable kernel function
            stabilizer: small positive stabilizer added to diagonal of P
        Returns:
            A tuple of optimal model parameters (a, a0) where a is n-by-1 and a0 is a scalar
    """    
    n, d = X.shape
    K = kernel_func(X, X)

    # construct P, q, G, and h for qp 
    q = np.concatenate([np.zeros(n + 1), np.ones(n)])
    G1 = np.hstack([np.zeros((n, n)), np.zeros((n, 1)), -np.eye(n)])
    G2 = np.hstack([-np.diag(y.reshape(n,)) @ K, -y, -np.eye(n)])
    G = np.vstack([G1, G2])
    h = np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))])

    P = np.zeros((n + 1 + n, n + 1 + n))
    P[:n, :n] = lamb * K  # only K is regularized
    P = P + stabilizer * np.eye(2*n + 1)  # add stabilizer

    # Convert to cvx matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u = np.array(sol['x']).flatten()
    a = u[:n][:, None] # make it nx1
    a0 = u[n] 

    return  a, a0

# Question 2 (c)
def adjClassify(Xtest, a, a0, X, kernel_func):
    """
        This function computes the prediction of a given model.
        Args:
            Xtest: mxd test matrix 
            a: nx1 model parameters
            a0: scalar intercept for model
            X: nxd training input matrix
            kernel_func: callable kernel function
        Returns:
            A mx1 vector of predictions for the given model 
    """
    y_hat = np.sign(kernel_func(Xtest, X) @ a + a0)
    return y_hat


# Question 2 (d)
def synExperimentsKernel():
    """
        This function runs a synthetic experiment for the adjExpLinear and adjHinge, calculating the average training and test accuracies 
        Returns:
            A 5x6 matrix of average training accuracies and a 5x6 matrix of average test accuracies 
    """
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    
    np.random.seed(51)
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel)
                train_acc_explinear[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) 
                test_acc_explinear[i, j, r] = np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel)) 

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) 
                test_acc_hinge[i, j, r] = np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel)) 

    # Calculate avg training acc for explinear & hinge loss
    avg_train_acc_explinear = np.mean(train_acc_explinear, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_train_acc = np.hstack([avg_train_acc_explinear, avg_train_acc_hinge])
    
    # Calculate avg test acc for explinear & hinge loss
    avg_test_acc_explinear = np.mean(test_acc_explinear, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)
    avg_test_acc = np.hstack([avg_test_acc_explinear, avg_test_acc_hinge])

    return avg_train_acc, avg_test_acc

if __name__ == "__main__":

    # Question 1 (d)
    # avg_train_acc, avg_test_acc = synExperimentsRegularize()

    # Question 2 (a)
    '''
    # Checking correctness by comparing to q1 (a) 
    n = 100
    lamb = 0.01
    gen_model = 1
    kernel_func = lambda X1, X2: linearKernel(X1, X2)

	# Generate data
    Xtrain, ytrain = generateData(n=n, gen_model=gen_model)

    a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel_func)
    w, w0 = minExpLinear(Xtrain, ytrain, lamb)

    K = kernel_func(Xtrain, Xtrain)
    exp_linear = Xtrain @ w + w0
    adj_linear = K @ a + a0

    print(f"EXPLINEAR:\n {exp_linear}")
    print(f"ADJEXPLINEAR:\n {adj_linear}")
    '''

    # Question 2 (b)
    '''
    # Checking correctness by comparing to q1 (b) 
    w, w0 = minHinge(Xtrain, ytrain, lamb)
    a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)
    
    hinge = Xtrain @ w + w0
    adj_hinge = K @ a + a0

    print(f"HINGE:\n {hinge}")
    print(f"ADJHINGE:\n {adj_hinge}")
    '''
    
    # Question 2 (d)
    avg_train_acc, avg_test_acc = synExperimentsKernel()
    print(f"Average TRAIN accuracy:\n{avg_train_acc}")
    print(f"Average TEST accuracy:\n{avg_test_acc}")


import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import *
import pandas as pd

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


# Question 1 (b)
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

    return avg_train_acc, avg_test_acc


# Question 2: Binary Classification (Adjoint Form)
# Question 2 (a)
def adjExpLinear(X, y, lamb, kernel_func):
    """
        This function computes the solution of the adjoint form of the regularized ExpLinear loss
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
            kernel_func: a callable kernel function
        Returns:
            A tuple of optimal model parameters (a, a0) where a is n-by-1 and a0 is a scalar
    """
    n, _ = X.shape

    # Define objective
    def obj_func(u):

        a0 = u[-1]  # last unknown is a0
        a = u[:-1]  # the first n dimensions are a
        a = a[:, None]  # make it n-by-1
        
        K = kernel_func(X, X) # kernel matrix
        m = y * (K.T @ a + a0) # margin

        mask = (m <= 0) # our condition
        result = np.empty_like(m)
        result[mask] = 1 - m[mask] # values <=0 are set to 1-m
        result[~mask] = np.exp(-m[~mask]) # values > 0 are set to e^{-m}

        loss = np.sum(result) # compute the loss
        reg = 0.5 * lamb * float((a.T @ K @ a)[0][0])
        
        return loss + reg

    # Initial guess of unknowns
    u0 = np.ones(n + 1)  

    sol = minimize(obj_func, u0)  # objective function + initial guess as inputs

    # Get the solution
    a = sol['x'][:-1][:, None]  # make it n-by-1
    a0 = sol['x'][-1]

    return a, a0

# Question 2 (b)
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    """
        This function computes the solution of the adjoint form of the regularized Hinge loss
        Args:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter (> 0)
            kernel_func: a callable kernel function
            stabilizer: small positive stabilizer added to diagonal of P
        Returns:
            A tuple of optimal model parameters (a, a0) where a is n-by-1 and a0 is a scalar
    """    
    n, d = X.shape
    K = kernel_func(X, X)

    # construct P, q, G, and h for qp 
    q = np.concatenate([np.zeros(n + 1), np.ones(n)])
    G1 = np.hstack([np.zeros((n, n)), np.zeros((n, 1)), -np.eye(n)])
    G2 = np.hstack([-np.diag(y.reshape(n,)) @ K, -y, -np.eye(n)])
    G = np.vstack([G1, G2])
    h = np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))])

    P = np.zeros((n + 1 + n, n + 1 + n))
    P[:n, :n] = lamb * K  # only K is regularized
    P = P + stabilizer * np.eye(2*n + 1)  # add stabilizer

    # Convert to cvx matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u = np.array(sol['x']).flatten()
    a = u[:n][:, None] # make it nx1
    a0 = u[n] 

    return  a, a0

# Question 2 (c)
def adjClassify(Xtest, a, a0, X, kernel_func):
    """
        This function computes the prediction of a given model.
        Args:
            Xtest: mxd test matrix 
            a: nx1 model parameters
            a0: scalar intercept for model
            X: nxd training input matrix
            kernel_func: callable kernel function
        Returns:
            A mx1 vector of predictions for the given model 
    """
    y_hat = np.sign(kernel_func(Xtest, X) @ a + a0)
    return y_hat


# Question 2 (d)
def synExperimentsKernel():
    """
        This function runs a synthetic experiment for the adjExpLinear and adjHinge, calculating the average training and test accuracies 
        Returns:
            A 5x6 matrix of average training accuracies and a 5x6 matrix of average test accuracies 
    """
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
        lambda X1, X2: polyKernel(X1, X2, 2),
        lambda X1, X2: polyKernel(X1, X2, 3),
        lambda X1, X2: gaussKernel(X1, X2, 1.0),
        lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    
    np.random.seed(51)
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel)
                train_acc_explinear[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) 
                test_acc_explinear[i, j, r] = np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel)) 

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(ytrain == adjClassify(Xtrain, a, a0, Xtrain, kernel)) 
                test_acc_hinge[i, j, r] = np.mean(ytest == adjClassify(Xtest, a, a0, Xtrain, kernel)) 

    # Calculate avg training acc for explinear & hinge loss
    avg_train_acc_explinear = np.mean(train_acc_explinear, axis=2)
    avg_train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    avg_train_acc = np.hstack([avg_train_acc_explinear, avg_train_acc_hinge])
    
    # Calculate avg test acc for explinear & hinge loss
    avg_test_acc_explinear = np.mean(test_acc_explinear, axis=2)
    avg_test_acc_hinge = np.mean(test_acc_hinge, axis=2)
    avg_test_acc = np.hstack([avg_test_acc_explinear, avg_test_acc_hinge])

    return avg_train_acc, avg_test_acc

if __name__ == "__main__":

    # Question 1 (d)
    # avg_train_acc, avg_test_acc = synExperimentsRegularize()

    # Question 2 (a)
    '''
    # Checking correctness by comparing to q1 (a) 
    n = 100
    lamb = 0.01
    gen_model = 1
    kernel_func = lambda X1, X2: linearKernel(X1, X2)

	# Generate data
    Xtrain, ytrain = generateData(n=n, gen_model=gen_model)

    a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel_func)
    w, w0 = minExpLinear(Xtrain, ytrain, lamb)

    K = kernel_func(Xtrain, Xtrain)
    exp_linear = Xtrain @ w + w0
    adj_linear = K @ a + a0

    print(f"EXPLINEAR:\n {exp_linear}")
    print(f"ADJEXPLINEAR:\n {adj_linear}")
    '''

    # Question 2 (b)
    '''
    # Checking correctness by comparing to q1 (b) 
    w, w0 = minHinge(Xtrain, ytrain, lamb)
    a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)
    
    hinge = Xtrain @ w + w0
    adj_hinge = K @ a + a0

    print(f"HINGE:\n {hinge}")
    print(f"ADJHINGE:\n {adj_hinge}")
    '''
    
    # Question 2 (d)
    avg_train_acc, avg_test_acc = synExperimentsKernel()
    print(f"Average TRAIN accuracy:\n{avg_train_acc}")
    print(f"Average TEST accuracy:\n{avg_test_acc}")




#Question 3 (a)
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    '''
    This function computes the best binary classifier using the dual form of the
    Support Vector Machine with hinge loss and L2 regularization.

    Args:
    X: nxd input matrix
    y: nx1 target vector containing class label in {-1,1}
    lamb: regularization hyper parameter
    kernel_func: callable kernel function
    stabilizer: small positive scalar added to the diagonal of P for numerical stability

    Returns:
        A tuple of optimal model parameters (a, b) where:
            a is an n-by-1 vector of dual weights (α*)
            b is a scalar intercept term
    '''



    n = X.shape[0]
    X = X.astype(float)
    y = y.astype(float)
    K = kernel_func(X, X).astype(float)
    Y = np.diagflat(y.ravel())

    #prepare variables to use the solver function
    G_upper = np.eye(n)
    G_lower = -np.eye(n)
    G = np.vstack((G_upper, G_lower)).astype(float)
    h = np.vstack((np.ones((n, 1)), np.zeros((n, 1)))).astype(float)
    P = (1.0 / lamb) * (Y @ K @ Y)
    P += stabilizer * np.eye(n)
    P = 0.5 * (P + P.T)
    P = P.astype(float) 
    q = -np.ones((n, 1), dtype=float)
    A = y.reshape(1, n).astype(float)
    b_eq = np.array([[0.0]], dtype=float)

    # Solve QP problem
    solved = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b_eq))
    a = np.array(solved['x']).reshape(n, 1)

    # Intercept term b calculations

    tol = 1e-5
    y_a = y * a
    dec = (1.0 / lamb) * (K @ y_a)              

    # margin SVs: 0 < alpha < 1
    mask_margin = (a > tol) & (a < 1 - tol)
    idx_margin = np.where(mask_margin.ravel())[0]

    if idx_margin.size > 0:
        vals = (y[idx_margin, 0] - dec[idx_margin, 0])
        b = float(np.median(vals))             
    else:
        # fall back to all support vectors alpha > 0
        idx_sv = np.where((a > tol).ravel())[0]
        if idx_sv.size > 0:
            vals = (y[idx_sv, 0] - dec[idx_sv, 0])
            b = float(np.median(vals))
        else:
            # extreme fallback: pick the largest alpha
            idx = int(np.argmax(a))
            b = float(y[idx, 0] - dec[idx, 0])

    return a, b

#Question 3 (b)
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    """
        This function computes predictions for test points using the dual SVM model.
        Args:
            Xtest: mxd test matrix
            a: nx1 vector of dual weights (α*)
            b: scalar intercept term
            X: nxd training input matrix
            y: nx1 training label vector in {-1, +1}
            lamb: regularization hyper-parameter (> 0)
            kernel_func: callable kernel function
        Returns:
            A mx1 vector of predictions yhat in {-1, +1}
    """
    yhat = np.sign((1.0/lamb)*(kernel_func(Xtest,X)@(y *a) )+b)
    return yhat

#Question 3 (c)
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    """
    This function performs k-fold cross-validation on the 4 and 9 dataset
    to evaluate different regularization strengths and kernel functions using
    the dual SVM model.

    Args:
        dataset_folder: absolute path to the dataset folder containing A2train.csv
        lamb_list: list of regularization hyper-parameters to test
        kernel_list: list of callable kernel functions to evaluate
        k: number of cross-validation folds 

    Returns:
        avg_acc: length of lamb list by length of kernel list matrix of average validation accuracies
        best_lamb: λ value that achieved the highest validation accuracy
        best_kernel: kernel function that achieved the highest validation accuracy
    """
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])
    np.random.seed(74)
    # Shuffle and divide the dataset into k folds for cross validation
    n = X.shape[0]
    shuffle =np.random.permutation(n)
    folds = np.array_split(shuffle, k)
    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                # Split data into training and validation sets
                val_index = folds[l]
                train_index = np.concatenate([folds[m] for m in range(k) if m != l])

                Xtrain = X[train_index] 
                ytrain = y[train_index] 
                Xval = X[val_index] 
                yval = y[val_index] 
                # Train and evaluate SVM model for this fold
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                # Compute validation accuracy for this fold
                cv_acc[l, i, j] = np.mean((yhat * yval) > 0)
    # Compute average accuracy across folds
    avg_acc = np.mean(cv_acc, axis=0)
    # Find best hyperparameter and kernel
    best_flat = np.argmax(avg_acc)
    best_i, best_j = np.unravel_index(best_flat, avg_acc.shape)
    best_lamb = lamb_list[best_i]
    best_kernel = kernel_list[best_j]

    return avg_acc, best_lamb, best_kernel
                    






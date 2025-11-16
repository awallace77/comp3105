import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigh
from cvxopt import matrix, solvers
from A3helpers import *
import pandas as pd
import random as rand

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
        
        scores = X @ W
        first_term = np.sum(logsumexp(scores, axis=1))
        second_term = np.sum(Y * scores)
        '''
        weights_inputs = (X @ W).T 
        labels_weights_inputs = (Y @ W.T @ X.T)

        first_term = np.sum(logsumexp(weights_inputs, axis=0))
        second_term = np.trace(labels_weights_inputs)
        '''
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

    Z = Xtest @ W          # (m,k)
    m, k = Z.shape

    # Get index with largest value
    i = np.argmax(Z, axis=1)

    # Set columns with largest values = 1
    Yhat = np.zeros((m, k), dtype=int)
    Yhat[np.arange(len(i)), i] = 1

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
    
    # Center with the mean
    X = X - np.mean(X, axis=0)

    # Get eigen vectors corresponding to the k largest eigen values (in ascending order)
    _, eigen_vectors = eigh(X.T @ X)
    eigen_vectors = eigen_vectors[:, -k:]
    
    # Each row is an eigen vector (k-by-d) in descending order
    U = np.flip(eigen_vectors.T, axis=0)

    return U

def projPCA(Xtest, mu, U):
    """
        Calculates the projected features of Xtest onto the direction U
        Args:
            Xtest: m-by-d input matrix
            mu: d-by-1 training mean vector
            U: k-by-d projection matrix
        Returns:
            m-by-k projected matrix that consists of the projected features onto the direction U
    """

    # Projection of a data points X is given by (X - mu.T) @ U.T
    Xproj = (Xtest - mu.T) @ U.T
    return Xproj

def kernelPCA(X, k, kernel_func):
    """
        Calculates the top k projecting vectors with proper normalization for dimensionality reduction
        Args:
            X: n-by-d input matrix
            k: scalar integer 1 <= k <= d
            kernel_func: a callable kernel function
        Returns:
            k-by-n matrix of coefficients whose rows correspond to the top-k coefficients with the largest variances
    """
    n, _ = X.shape

    X = X.astype(float) # convert to float to avoid overflow

    # Calculate kernel
    K = kernel_func(X, X)

    # Center kernel
    ones = np.ones((n,n))
    K_centered = K - ((1/n) * ones @ K) - ((1/n) * K @ ones) + ((1/(n**2)) * ones @ K @ ones)

    # Get eigen vectors corresponding to the k largest eigen values (in ascending order)
    eigen_values, eigen_vectors = eigh(K_centered)
    eigen_values= eigen_values[-k:]
    eigen_vectors = eigen_vectors[:, -k:]

    # Rescale by 1/lambda n
    A = eigen_vectors / np.sqrt(eigen_values)

    # Each row is an eigen vector (k-by-n) in descending order
    A = np.flip(A.T, axis=0)

    return A

def projKernelPCA(Xtest, Xtrain, kernel_func, A):
    """
        Calculates the projection of Xtest onto the directions specified by A
        Args:
            Xtest: m-by-d input matrix
            Xtrain: n-by-d input matrix
            kernel_func: callable kernel function
            A: k-by-n coefficient matrix
        Returns:
            m-by-k projected matrix Xproj consisting of the projected features of Xtest onto the directions specified by A
    """
    og_m = Xtest.shape[0]
    og_n = Xtrain.shape[0]

    # Construct kernels
    K_te_tr = kernel_func(Xtest, Xtrain)
    K_tr_tr = kernel_func(Xtrain, Xtrain)
    
    m = K_te_tr.shape[0]
    n = K_tr_tr.shape[0]

    assert(og_m == m and og_n == n)
    
    # Center kernel
    ones_n_n = np.ones((n,n))
    ones_m_n = np.ones((m,n))
    K_te_tr_c = K_te_tr - (1/n * ones_m_n @ K_tr_tr) - (1/n * K_te_tr @ ones_n_n) + (1/n**2 * ones_m_n @ K_tr_tr @ ones_n_n)

    # Project values
    Xproj = K_te_tr_c @ A.T
    return Xproj

def synClsExperimentsPCA():
    """
        Runs a synthetic PCA experiment
        Returns:
            2-by-2 matrix of average training accuracies and a 2-by-2 matrix of average test accuracies
    """

    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])

    # DONE: Change the following random seed to your GROUP number (<=3digits)
    np.random.seed(51)

    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)
                U = PCA(Xtrain, k)
                Xtrain_proj = projPCA(Xtrain, np.mean(Xtrain, axis=0)[:, None], U) # DONE: call your projPCA to find the new features
                Xtest_proj = projPCA(Xtest, np.mean(Xtest, axis=0)[:, None], U) # DONE: call your projPCA to find the new features
                Xtrain_proj = augmentX(Xtrain_proj) # add augmentation
                Xtest_proj = augmentX(Xtest_proj)
                W = minMulDev(Xtrain_proj, Ytrain) # from Q1
                Yhat = classify(Xtrain_proj, W) # from Q1
                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1
                Yhat = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)

    # compute the average accuracies over runs
    train_acc = np.mean(train_acc, axis=2)
    test_acc  = np.mean(test_acc, axis=2)

    # return 2-by-2 train accuracy and 2-by-2 test accuracy
    return train_acc, test_acc


# Question 3a
def kmeans(X, k, max_iter=1000):
    """
    Runs the standard k means algorithm.

    Args:
        X: input data matrix
        k: scalar integer specifying the number of clusters
        max_iter: maximum number of iterations to perform

    Returns:
        Y: assignment data point to cluster matrix 
        U: matrix of cluster centers
        obj_val: scalar objective value corresponding to the final clustering
    """
    n, d = X.shape
    assert max_iter > 0 and k < n
    # U is found by picking k random points from X to be the center points
    U = X[np.random.choice(X.shape[0], k, replace=False), :]

    for i in range(max_iter):
        Y = np.zeros((n,k))
        D = np.zeros((n,k))
        
        # Caclulate Y by looking for the closest U center for each X data point
        for x in range(X.shape[0]):
            min_D = np.inf
            min_cluster = -1
            for j in range(U.shape[0]):
                D[x, j] = np.linalg.norm(X[x] - U[j])
                if D[x,j] < min_D: 
                    min_cluster = j
                    min_D = D[x,j]
            Y[x] = 0
            Y[x,min_cluster] = 1 #Find the new cluster assignments

        old_U = U
        #find the mean of all the clusters and that is new cluster center for Uj
        U = np.zeros((k,d))
        for j in range(k):
            cluster_points = X[Y[:,j]==1]
            
            if cluster_points.shape[0] > 0:
                U[j] = cluster_points.mean(axis=0)
            else: #if cluster has no points:
                U[j] = old_U[j]   # keep old center


        if np.allclose(old_U, U):
            break
    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, U, obj_val

# Question 3b 
def repeatKmeans(X, k, n_runs=100):
    """
    Repeats k means multiple times with different random initializations and returns the clustering with the smallest objective value

    Args:
        X: input data matrix
        k: scalar integer specifying the number of clusters
        n_runs: number of times to re run k means with different intial cluster centers

    Returns:
        best_Y: assignment data point to cluster matrix with the smallest k mean objective value
        best_U: matrix of cluster centers coresponding to best_Y
        best_obj_val: smallest scalar objective value achieved over all runs
    """
    best_obj_val = float('inf')
    best_Y = None
    best_U = None
    for r in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        #if obj_val is smallest then keep that one
        if obj_val<best_obj_val:
            best_obj_val = obj_val
            best_Y = Y
            best_U = U
    #return Y and U of smallest Object Value
    return(best_Y,best_U,best_obj_val)

#Question 3c
def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
    """
    Tests the k means fucntion for a list of candidate k values using repeated k means for each k.

    Args:
        X: input data matrix.
        k_candidates: list of candidate integers for the number of clusters.

    Returns:
        list_obj_val: list of scalar objective values, where the ith entry is the best k means objective value obtained for
        k = k_candidates[i].
    """
    list_obj_val = []
    for k in k_candidates:
        Y, U, obj_val = repeatKmeans(X,k)
        list_obj_val.append(obj_val)
    return(list_obj_val)




# Question 3d
def kernelKmeans(X, kernel_func, k, init_Y, max_iter=1000):
    """
    Runs kernel k means given an initial cluster assignment.

    Args:
        X: input data matrix.
        kernel_func: callable kernel function that takes two matrices (X1, X2) and returns the n1 by n2 kernel matrix K(X1, X2)
        k: scalar integer specifying the number of clusters
        init_Y: initial assignment data point to cluster matrix 
        max_iter: maximum number of kernel k means iterations to perform

    Returns:
        Y: assignment data point to cluster matrix 
        obj_val: scalar objective value corresponding to the final clustering
    """
    n, d = X.shape
    assert max_iter > 0 and k < n

    #compute the kernel matrix
    K = kernel_func(X, X)
    Y = init_Y


    #the diagonal of K
    Kii = np.diag(K)

    for i in range(max_iter):
        D = np.zeros((n,k))

        #compute the distances for each cluster j
        for j in range(k):
            #indexs of the points curretly assigned to this cluster
            idx_j = np.where(Y[:,j] == 1)[0]
            n_j = len(idx_j)


            #just incase cluster is empty, we set distance to infinite so it wont be chosen
            if n_j == 0:
                D[: , j] = np.inf
                continue
            
            #this is the kernel k-mean distance formula
            D[:, j] = Kii - (2.0/n_j)*np.sum(K[:,idx_j], axis = 1) + (1.0/n_j**2)*np.sum(K[np.ix_(idx_j, idx_j)])
        
        old_Y = Y.copy()

        min_cluster = np.argmin(D, axis=1)

        Y = np.zeros((n,k))
        Y[np.arange(n),min_cluster] = 1

        if np.allclose(old_Y, Y):
            break

    obj_val = (0.5 / n) * np.sum(D.min(axis=1))
    return Y, obj_val

if __name__ == "__main__":

    # Question 1 testing
    print("QUESTION 1")
    X = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [3.0, 1.0]
    ]) 

    Y = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]) 

    W = minMulDev(X, Y)
    Yhat = classify(X, W)
    print(f"Accuracy: {calculateAcc(Yhat, Y)}")

    # Question 2a testing
    print("QUESTION 2A")
    data_matrix = np.loadtxt('./a3/src/A3train.csv', delimiter=',')
    n, d = data_matrix.shape
    U = PCA(data_matrix, d-2)
    plotImgs(U)

    # Question 2c testing
    print("QUESTION 2C")
    X = np.array([
        [1, 2, 1],
        [0, 2, 3],
        [1, 3, 2]
    ])

    K = linearKernel(X, X)
    print(f"K:\n{K}")
    U = PCA(X, 2)
    A = kernelPCA(X, 2, linearKernel)
    print(f"Final result PCA:\n{U}")
    print(f"Final result k_PCA:\n{A}")
    X_centered = X - np.mean(X, axis=0)
    print(f"U = AX\n{U} = \n{A @ X_centered}")
    print("\n")

    # Question 2d testing
    print("QUESTION 2D")
    Xtest = np.array([
        [0, 1, 3],
        [1, 3, 0],
        [3, 2, 4]
    ])
    mu = np.mean(X, axis=0)
    Xtrain_centered = X - np.mean(X, axis=0)
    Xtest_centered = Xtest - np.mean(X, axis=0) 
    Xproj_PCA = projPCA(Xtest, mu.reshape(-1,1), U)
    Xproj_KPCA = projKernelPCA(Xtest, X, linearKernel, A)

    print(f"INNER PRODUCT:\n{Xtest_centered @ Xtrain_centered.T @ A.T}")
    print(f"projPCA:\n{Xproj_PCA}")
    print(f"projKernelPCA:\n{Xproj_KPCA}")
    
    # Question 2e testing
    print("QUESTION 2E")
    train_acc, test_acc = synClsExperimentsPCA()
    print(f"train_acc:\n{train_acc}")
    print(f"test_acc:\n{test_acc}")

    # Question 3c testing
    print("QUESTION 3C")
    Xtrain, Ytrain = generateData(n=100, gen_model=2)
    obj_val_list = chooseK(Xtrain)
    print(f"test_acc:\n{obj_val_list}")


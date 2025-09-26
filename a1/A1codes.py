
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import os
from scipy.optimize import minimize
import autograd as ag
import autograd.numpy as agnp

# HELPER FUNCTIONS ***************************************************

def L2_loss(X, w, y):
    """
        Calculates the L2 loss of given inputs, model, and labels
        Parameters:
            X: a nxd matrix of inputs
            w: a dx1 vector of weights
            y: a nx1 vector of targets
        Returns: the computed L2 loss
    """
    # 1/2n (Xw - y)^2
    return (1/2) * np.mean(((X @ w) - y) ** 2)

def Linf_loss(X, w, y): 
    """
        Calculates the Linf loss for a given inputs, model, and labels
        Parameters:
            X: a nxd matrix of inputs
            w: a dx1 vector of weights
            y: a nx1 vector of targets
        Returns: the computed Linf loss
    """
    # max_{i = 1, ..., n} |x_i^T w - y_i|
    return np.max(np.abs((X @ w) - y))

def evaluate_L2_Linf(loss, run, X, w_L2, w_Linf, y):
    """
        Calculates the L2 and Linf losses for a given L2 and Linf model
        Parameters:
            loss: the 2x2 loss matrix (by reference)
            run: the current run (number)
            w_L2: the L2 model 
            w_Linf: the Linf model
            y: the truth vector
        Returns: the updated loss matrix
    """
    loss[run][0][0] = L2_loss(X, w_L2, y) 
    loss[run][0][1] = Linf_loss(X, w_L2, y)
    loss[run][1][0] = L2_loss(X, w_Linf, y)
    loss[run][1][1] = Linf_loss(X, w_Linf, y) 


def avg_L2_Linf_loss(loss, n_runs):
    """
        Calculates the average loss in n runs
        Parameters:
            loss: the 2x2 model x loss matrix 
            n_runs: the total number of runs for your experiment
        Returns: the averge L2 and Linf loss
    """
    avg_loss = np.zeros([2,2])

    # Total L2 model - L2 losses and Linf losses
    total_L2model_L2_loss = np.sum(loss[:, 0, 0]) 
    total_L2model_Linf_loss = np.sum(loss[:, 0, 1]) 

    # Total Linf model - L2 losses and Linf losses
    total_Linfmodel_L2_loss = np.sum(loss[:, 1, 0])
    total_Linfmodel_Linf_loss = np.sum(loss[:, 1, 1]) 
    
    avg_loss[0][0] = total_L2model_L2_loss / n_runs
    avg_loss[0][1] = total_L2model_Linf_loss / n_runs   
    avg_loss[1][0] = total_Linfmodel_L2_loss / n_runs
    avg_loss[1][1] = total_Linfmodel_Linf_loss / n_runs 

    return avg_loss


# QUESTION 1 ***************************************************

def minimizeL2(X, y):
    """
        Calculates the parameters w corresponding to the solution of the L2 losses
        Parameters:
            X: a nxd matrix 
            y: a nx1 target/label vector
        Returns: dx1 vector of parameters w corresponding to the solution of the L2 losses
    """
    # w = (X.T X)^{-1} X.T y
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def minimizeLinf(X, y):
    """
        Calculates the parameters w corresponding to the solution of the Linf losses
        Parameters: 
            X: a nxd matrix 
            y: a nx1 target/label vector
        Returns: dx1 vector of parameters w corresponding to the solution of the Linf loss
    """
    # cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])
    
    # nxd dimensions
    n, d = X.shape

    # Objective coefficients
    c = np.concatenate([np.zeros([d, 1]), np.ones([1,1])])

    # Constraint matrix
    G_1 = np.concatenate([np.zeros([1, d]), -np.ones([1,1])], axis=1)
    G_2 = np.concatenate([X, -(np.ones([n, 1]))], axis=1)
    G_3 = np.concatenate([-X, -(np.ones([n, 1]))], axis=1)
    G = np.concatenate([G_1, G_2, G_3], axis=0)

    # Right hand side vector
    h = np.concatenate([np.zeros((1, 1)), y, -y], axis=0)

    # Convert to solver matrix
    c_final = matrix(c)
    G_final = matrix(G)
    h_final = matrix(h)

    # Solve for unknowns
    solvers.options['show_progress'] = False
    sol = solvers.lp(c_final, G_final, h_final)

    # Return only values for w and do not include threshold \delta
    final_sol = np.array(sol['x'])[:-1, :]
    return final_sol


def synRegExperiments():
    """
        Calculates the average training and test losses in 100 runs for L2 & Linf models
        Returns: A 2x2 matrix of average traning losses and a 2x2 matrix of average test losses for L2 & Linf models and L2 & Linf losses
    """
    def genRealData(n_points, is_training=False):
        """
        This function generates real data from regression_train and regression_test csv
        """
        X = []
        # Trying with regression training data
        try:
            if(is_training):
                X = np.genfromtxt('a1/toy_data/regression_train.csv', delimiter=',', skip_header=1).astype(float)
            else: 
                X = np.genfromtxt('a1/toy_data/regression_test.csv', delimiter=',', skip_header=1).astype(float)
        except FileNotFoundError:
            print("Error: file not found. Please create the file or provide the correct path.")
        except Exception as e:
            print(f"An error occurred: {e}")

        y = X[:, -1].reshape(X.shape[0], 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # augment input
        X = np.delete(X, X.shape[1] - 1, 1)

        return X, y

    def genData(n_points, is_training=False):
        """
        This function generate synthetic data
        """
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        if is_training:
            y[0] *= -0.1

        return X, y

    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    
    np.random.seed(101210291)
    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)

        # For data from regression_train.csv and regression_test.csv
        # Xtrain, ytrain = genRealData(n_train, is_training=True)
        # Xtest, ytest = genRealData(n_test, is_training=False)

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # TRAINING DATA: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the training
        # data). Save them to `train_loss`
        evaluate_L2_Linf(train_loss, r, Xtrain, w_L2, w_Linf, ytrain)
        
        # TEST DATA: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the test
        # data). Save them to `test_loss`
        evaluate_L2_Linf(test_loss, r, Xtest, w_L2, w_Linf, ytest)

    # TRAINING DATA - compute the average losses over runs
    avg_train_loss = avg_L2_Linf_loss(train_loss, n_runs)
    
    # TEST DATA - compute the average losses over runs
    avg_test_loss = avg_L2_Linf_loss(test_loss, n_runs)
    
    # Return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return avg_train_loss, avg_test_loss


def preprocessCCS(dataset_folder):
    """
        Preprocesses data
        Parameters:
            dataset_folder: the absolute path of the CCS dataset folder
        Returns: the preprocessed nxd input matrix X and an nx1 label vector y
    """
    filepath = os.path.join(dataset_folder,"Concrete_Data.xls")
    file = pd.read_excel(filepath)
    X = file.iloc[:,:-1].to_numpy(dtype=float) #all columns except for last
    y = file.iloc[:,-1:].to_numpy(dtype=float).reshape(-1,1) # only the last column

    return X,y


def runCCS(dataset_folder):
    """
        Calculates the training and test losses for a given dataset folder
        Parameters:
            dataset_folder: the absolute path of the CCS dataset folder
        Output: the training and test losses
    """
    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    n_runs = 100
    train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics

    np.random.seed(101260693)
    for r in range(n_runs):
        # Randomly partition the dataset into two parts (50%
        # training and 50% test)
        n = X.shape[0]
        
        permed = np.random.permutation(n) # returns a random reorder of the number of data points

        split = n//2 #split at the half way 

        idx_fh = permed[:split]#this includes everything before the split
        idx_sh = permed[split:]#this includes everything after the split

        Xfh, yfh = X[idx_fh], y[idx_fh]
        Xsh, ysh = X[idx_sh], y[idx_sh]

        # Learn two different models from the training data
        # using L2 and L infinity losses

        modell2 = minimizeL2(Xfh,yfh)
        modellinf = minimizeLinf(Xfh,yfh)

        # Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the training
        # data). Save them to `train_loss`
        avg_train_loss = np.zeros([2,2])

        # L2 model
        L2model_L2_loss = L2_loss(Xfh, modell2, yfh) 
        L2model_Linf_loss = Linf_loss(Xfh, modell2, yfh)
        train_loss[r][0][0] = L2model_L2_loss
        train_loss[r][0][1] = L2model_Linf_loss
        # L_inf model
        Linfmodel_L2_loss = L2_loss(Xfh, modellinf, yfh) 
        Linfmodel_Linf_loss = Linf_loss(Xfh, modellinf, yfh)
        train_loss[r][1][0] = Linfmodel_L2_loss
        train_loss[r][1][1] = Linfmodel_Linf_loss


        # Done: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the test
        # data). Save them to `test_loss`
        L2model_L2_loss = L2_loss(Xsh, modell2, ysh) 
        L2model_Linf_loss = Linf_loss(Xsh, modell2, ysh)
        test_loss[r][0][0] = L2model_L2_loss
        test_loss[r][0][1] = L2model_Linf_loss
        # L_inf model
        Linfmodel_L2_loss = L2_loss(Xsh, modellinf, ysh) 
        Linfmodel_Linf_loss = Linf_loss(Xsh, modellinf, ysh)
        test_loss[r][1][0] = Linfmodel_L2_loss
        test_loss[r][1][1] = Linfmodel_Linf_loss

    # compute the average losses over runs

    # TRAINING DATA - compute the average losses over runs
    avg_train_loss = np.zeros([2,2])

    # Total L2 model - L2 losses and Linf losses
    total_L2model_L2_loss = np.sum(train_loss[:, 0, 0]) 
    total_L2model_Linf_loss = np.sum(train_loss[:, 0, 1]) 

    # Total Linf model - L2 losses and Linf losses
    total_Linfmodel_L2_loss = np.sum(train_loss[:, 1, 0]) 
    total_Linfmodel_Linf_loss = np.sum(train_loss[:, 1, 1]) 
    
    avg_train_loss[0][0] = total_L2model_L2_loss / n_runs
    avg_train_loss[0][1] = total_L2model_Linf_loss / n_runs   
    avg_train_loss[1][0] = total_Linfmodel_L2_loss / n_runs
    avg_train_loss[1][1] = total_Linfmodel_Linf_loss / n_runs 
    
    # TEST DATA - compute the average losses over runs
    avg_test_loss = np.zeros([2,2])
    
    # Total L2model L2 losses and Linf losses
    total_L2model_L2_loss = np.sum(test_loss[:, 0, 0]) 
    total_L2model_Linf_loss = np.sum(test_loss[:, 0, 1]) 

    # Total Linf Model L2 losses and Linf losses
    total_Linfmodel_L2_loss = np.sum(test_loss[:, 1, 0]) 
    total_Linfmodel_Linf_loss = np.sum(test_loss[:, 1, 1]) 
    
    avg_test_loss[0][0] = total_L2model_L2_loss / n_runs
    avg_test_loss[0][1] = total_L2model_Linf_loss / n_runs
    avg_test_loss[1][0] = total_Linfmodel_L2_loss / n_runs
    avg_test_loss[1][1] = total_Linfmodel_Linf_loss / n_runs


    # return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return avg_train_loss, avg_test_loss


""" QUESTION 2 ************************************************************"""

def linearRegL2Obj(w, X, y):
    """
        Calculates a scalar objective value
        Parameters:
            w: dx1 parameter vector
            X: nxd input matrix 
            y: nx1 label vector
        Ouput: A scalar value that is the objective value of 1/2n ||Xw - y||_{2}^{2}
    """
    n, _ = X.shape
    obj_value = (1 / (2*n)) * (((X @ w - y).T) @ (X @ w - y))[0][0]
    return obj_value


def linearRegL2Grad(w, X, y):
    """
        Calculates the vector gradient of the L2 objective function
        Parameters:
            w: dx1 parameter vector 
            X: nxd input matrix 
            y: nx1 label vector
        Returns: A vector gradient that is the analytic form gradient of size dx1
    """
    n, _ = X.shape
    obj_gradient = (1/n) * X.T @ (X @ w - y)
    return obj_gradient


def find_opt(obj_func, grad_func, X, y):
    """
        Finds the optimal solution of a convex optimization problem using the minimize from scipy.optimize
        Parameters:
            obj_func: An objective function
            grad_func: The gradient of the objective function 
            X: nxd input matrix
            y: nx1 label vector
        Returns: A dx1 parameter vector
    """
    d = X.shape[1]
    w_0 = np.random.rand(d)
    
    def func(w):
        w = w[:, None]
        obj_value = obj_func(w, X, y)
        return obj_value 
    
    def gd(w):
        w = w[:, None]
        grad = grad_func(w, X, y)[:, 0] # turn back into 1-d array
        return grad

    return minimize(func, w_0, jac=gd)['x'][:, None]
        

def sigmoid(z):
    """
        The sigmoid function
        Parameters:
            z: the bector to apply the sigmoid function to
        Returns: the value of apply the sigmoid function to z
    """
    return 1 / (1 + np.exp(-z))

# Done: Add DOCSTRING comments
def logisticRegObj(w, X, y):
    """
        Calculates the cross-entropy loss for binary logistic regression
        Parameters:
            w: a dx1 vector of parameters
            X: a nxd input matrix
            y: a nx1 binary label vector with values in {0, 1}
        Returns: a scalar loss value equal to (1/n) * [sum(log(1 + exp(Xw))) - y^T (Xw)]
    """
    n, _ = X.shape
    z = X @ w
    loss = (1/n) * (np.sum(np.logaddexp(0, z)) - np.sum(y * z))
    return float(loss)

def logisticRegGrad(w, X, y):
    """
        Calculates the gradient of the cross-entropy loss for binary logistic regression.
        Parameters:
            w: a dx1 vector of parameters
            X: a nxd input matrix
            y: a nx1 binary label vector with values in {0, 1}
        Returns: a dx1 gradient vector equal to (1/n) * X^T (sigmoid(Xw) - y)
    """
    n, d = X.shape

    z = (1/n)*(X.T @ (sigmoid(X@w)-y))
    return z


def synClsExperiments():
    """
        Calculates the average training accuracies and average test accuracies accross 100 runs
        Returns: 4x3 matrix train_acc of average training accuracies and a 4x3 matrix test_acc of average test accuracies
    """
    def genData(n_points, dim1, dim2):
        """
        This function generate synthetic data
        """
        c0 = np.ones([1, dim1]) # class 0 center
        c1 = -np.ones([1, dim1]) # class 1 center
        X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
        X0[:, :dim1] += c0
        X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
        X1[:, :dim1] += c1
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y 
    
    def genRealData(n_points, is_training=False):
        """
        This function generates real data from regression_train and regression_test csv
        """
        X = []
        # Trying with regression training data
        try:
            if(is_training):
                X = np.genfromtxt('a1/toy_data/classification_train.csv', delimiter=',', skip_header=1).astype(float)
            else: 
                X = np.genfromtxt('a1/toy_data/classification_test.csv', delimiter=',', skip_header=1).astype(float)
        except FileNotFoundError:
            print("Error: file not found. Please create the file or provide the correct path.")
        except Exception as e:
            print(f"An error occurred: {e}")

        y = X[:, -1].reshape(X.shape[0], 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # augment input
        X = np.delete(X, X.shape[1] - 1, 1)

        return X, y


    def runClsExp(m=100, dim1=2, dim2=2):
        """
        Run classification experiment with the specified arguments
        """
        n_test = 1000
        # Xtrain, ytrain = genData(m, dim1, dim2)
        # Xtest, ytest = genData(n_test, dim1, dim2)

        # To test w real data from classification_train.csv and classification_test.csv 
        Xtrain, ytrain = genRealData(m, True)
        Xtest, ytest = genRealData(m)

        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # Compute predicted labels of the training points
        ytrain_hat = np.where(Xtrain @ w_logit >= 0, 1, 0) 
        # Compute the accuarcy of the training set (1 - missclasification rate)
        train_acc = 1 - np.mean(np.where(ytrain_hat - ytrain != 0, 1, 0)) # 

        # Compute predicted labels of the test points
        ytest_hat = np.where(Xtest @ w_logit >= 0, 1, 0) 
        # Compute the accuarcy of the test set
        test_acc = 1 - np.mean(np.where(ytest_hat - ytest != 0, 1, 0)) 

        return train_acc, test_acc

    n_runs = 100
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])
    np.random.seed(101210291)
    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)
    
    # compute the average accuracies over runs
    avg_train_acc = np.mean(train_acc, axis=0)
    avg_test_acc = np.mean(test_acc, axis=0)  

    # ith row(s), jth inner row, kth option (m, dim1, dim2), 
    # avg = np.mean(train_acc[:,2,0])

    # return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable
    return avg_train_acc, avg_test_acc


#q2d 
# Done: Add DOCSTRINGS (function comments) - please reference other commented functions
def quad_expand(X):
    """
        Expands features by appending element wise sqaured terms.
        Parameters:
            X: a nxd input matrix
        Returns: a n x (2d) matrix formed by concatenating X with X**2 along the last axis
    """
    return np.concatenate([X, X**2], axis=1)

# Done: Add DOCSTRINGS (function comments) - please reference other commented functions
def preprocessBCW(dataset_folder):
    """
        Loads and preprocesses the Breast Cancer Wisconsin dataset.
        Parameters:
            dataset_folder: the absolute path of the BCW dataset folder
        Returns: the preprocessed nxd input matrix X and an nx1 label vector y with values in {0.0, 1.0}
    """
    filepath = os.path.join(dataset_folder,'wdbc.data')
    file = pd.read_csv(filepath, header=None)
    y = file.iloc[:, 1].map({'B': 0.0, 'M': 1.0}).to_numpy().reshape(-1, 1)
    X = file.iloc[:, 2:].to_numpy(dtype=float)
    return X, y

# Done: Add DOCSTRINGS (function comments) - please reference other commented functions
def runBCW(dataset_folder):
    """
        Trains and evaluates logistic regression on the BCW dataset.
        Parameters:
            dataset_folder: the absolute path of the BCW dataset folder
        Returns: the average training accuracy and the average test accuracy over the 100 runs
    """
    X, y = preprocessBCW(dataset_folder)
    X = quad_expand(X)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 100
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])
    # Change the following random seed to one of your student IDs
    np.random.seed(101260693)
    for r in range(n_runs):
        # Randomly partition the dataset into two parts (50%
        # training and 50% test) 
        n = X.shape[0]
        permed = np.random.permutation(n)
        split = n // 2
        idx_fh = permed[:split]
        idx_sh = permed[split:]

        Xfh, yfh= X[idx_fh], y[idx_fh]
        Xsh, ysh = X[idx_sh], y[idx_sh]
        
        w = find_opt(logisticRegObj, logisticRegGrad, Xfh, yfh)
        # Evaluate the model's accuracy on the training
        # data. Save it to `train_acc` //
        yhat_train = (sigmoid(Xfh @ w) >= 0.5).astype(float)
        train_acc[r] = 1.0 - np.mean(yhat_train != yfh)
        
        #  Evaluate the model's accuracy on the test
        # data. Save it to `test_acc`
        yhat_test  = (sigmoid(Xsh @ w) >= 0.5).astype(float)
        test_acc[r]  = 1.0 - np.mean(yhat_test  != ysh)

    # compute the average accuracies over runs
    avg_train_acc = float(np.mean(train_acc))
    avg_test_acc = float(np.mean(test_acc))
    # return two variables: the average training accuracy and average test accuracy
    return avg_train_acc, avg_test_acc


if __name__ == "__main__":

    # Question 1a ****************************************************
    # Please use a1testbed.py

    # Question 1b ****************************************************
    # Please see A1report.pdf
    
    # Question 1c ****************************************************
    print(f"\nQUESTION 1C: Linear regression synthetic experiments\n")
    train_loss, test_loss = synRegExperiments()
    print(f"TRAINING: \n{train_loss}\nTESTING:\n{test_loss}")
    print("\n")

    
    # Question 2a.1 ****************************************************
    # Analytic gradient
    print(f"\nQUESTION 2A.1: Linear regression L2 objective and gradient functions\n")
    X = agnp.random.randn(100, 3)              # n=100 samples, d=3 features
    w = agnp.random.randn(3).reshape(3, 1)     # parameter vector w (d,)
    y = agnp.random.randn(100).reshape(100, 1) # target vector (n,)
    
    # L2 loss w.r.t. w
    def L2_loss(w, X, y):
        return (1/2) * agnp.mean(((X @ w) - y) ** 2)

    # autograd automatically computes ∂L/∂w
    grad_L2 = ag.grad(L2_loss)  
    gradient = grad_L2(w, X, y)

    # Compare the gradients between autograd and our implementation
    print(f"Autograd gradient implementation: \n{gradient}")
    print(f"Our gradient implemenation: {linearRegL2Grad(w, X, y)}\n")
    print("\n")


    # Question 2a.2 ****************************************************
    print(f"\nQUESTION 2A.2: Finding optimal solution of a convex optimization problem for linear regression L2 objective function")
    w = find_opt(linearRegL2Obj, linearRegL2Grad, X, y)
    w_analytic = minimizeL2(X, y)
    print(f"via scipy.optimize minimize: \n {w}")
    print(f"via minimizeL2: \n {w_analytic}\n")
    
    
    # Question 2b ****************************************************
    print(f"\nQUESTION 2B: Logistic regression objective and gradient test\n")
    X = np.random.randn(20, 3)
    X = np.concatenate((np.ones((20,1)), X), axis=1)
    y = (np.random.rand(20,1) > 0.5).astype(float)

    w0 = np.random.randn(X.shape[1], 1)
    print(f"Objective at random w: {logisticRegObj(w0, X, y)}")
    print(f"Gradient shape: {logisticRegGrad(w0, X, y).shape}\n")


    # Question 2c: ****************************************************
    train_acc, test_acc = synClsExperiments()
    print(f"\nQUESTION 2C: Average train and test accuracies for different training sizes (10, 50, 100, 200) and dimensions (dim1=1,2,4,8 and dim2=1,2,4,8\n")
    print(f"TRAINING: \n{train_acc}\nTESTING:\n{test_acc}")
    print("\n")

    

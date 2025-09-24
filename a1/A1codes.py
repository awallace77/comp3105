
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import os
from scipy.optimize import minimize

'''
    L2_loss
    Input X: a nxd matrix of inputs
    Input w: a dx1 vector of weights
    Input y: a nx1 vector of targets
    Output: the computed L2 loss
'''
def L2_loss(X, w, y):
    # 1/2n (Xw - y)^2
    return (1/2) * np.mean(((X @ w) - y) ** 2)


'''
    Linf_loss
    Input X: a nxd matrix of inputs
    Input w: a dx1 vector of weights
    Input y: a nx1 vector of targets
    Output: the computed Linf loss
'''
def Linf_loss(X, w, y): 
    # max_{i = 1, ..., n} |x_i^T w - y_i|
    return np.max(np.abs((X @ w) - y))


'''
    minimizeL2
    Input X: a nxd matrix 
    Input y: a nx1 target/label vector
    Output: dx1 vector of parameters w corresponding to the solution of the L2 losses
    w = (X.T X)^{-1} X.T y
'''
def minimizeL2(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

'''
    minimizeLinf
    Input X: a nxd matrix 
    Input y: a nx1 target/label vector
    Output: dx1 vector of parameters w corresponding to the solution of the L_{\infty} loss
'''
def minimizeLinf(X, y):
    '''
    cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])
    '''
    
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

'''
    evaluate_L2_Linf
    Input loss: the 2x2 loss matrix (by reference)
    Input run: the current run (number)
    Input w_L2: the L2 model 
    Input w_Linf: the Linf model
    Input y: the truth vector
    Ouput: the updated loss matrix
'''
def evaluate_L2_Linf(loss, run, X, w_L2, w_Linf, y):
    loss[run][0][0] = L2_loss(X, w_L2, y) 
    loss[run][0][1] = Linf_loss(X, w_L2, y)
    loss[run][1][0] = L2_loss(X, w_Linf, y)
    loss[run][1][1] = Linf_loss(X, w_Linf, y) 

'''
    avg_L2_Linf_loss
    Input loss: the 2x2 model x loss matrix 
    Input n_runs: the total number of runs for your experiment
    Output: the averge L2 and Linf loss
'''
def avg_L2_Linf_loss(loss, n_runs):
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


'''
    synRegExperiments
    Output: A 2x2 matrix of average traning losses and a 2x2 matrix of average test losses for L2 & Linf models and L2 & Linf losses
'''
def synRegExperiments():

    def genRealData(n_points, is_training=False):
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
        '''
        This function generate synthetic data
        '''
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
    
    # DONE: Change the following random seed to one of your student IDs
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
    print(avg_train_loss)
    
    # TEST DATA - compute the average losses over runs
    avg_test_loss = avg_L2_Linf_loss(test_loss, n_runs)
    print(avg_test_loss)
    
    # Return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return avg_train_loss, avg_test_loss


if __name__ == "__main__":
    synRegExperiments()

def preprocessCCS(dataset_folder):
    filepath = os.path.join(dataset_folder,"Concrete_Data.xls")
    file = pd.read_excel(filepath)
    X = file.iloc[:,:-1].to_numpy(dtype=float) #all columns except for last
    y = file.iloc[:,-1:].to_numpy(dtype=float).reshape(-1,1) # only the last column

    return X,y

def runCCS(dataset_folder):
    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    n_runs = 100
    train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(101260693)
    for r in range(n_runs):
    # DONE: Randomly partition the dataset into two parts (50%
    # training and 50% test)
        n = X.shape[0]
        
        permed = np.random.permutation(n) # returns a random reorder of the number of data points

        split = n//2 #split at the half way 

        idx_fh = permed[:split]#this includes everything before the split
        idx_sh = permed[split:]#this includes everything after the split

        Xfh, yfh = X[idx_fh], y[idx_fh]
        Xsh, ysh = X[idx_sh], y[idx_sh]





    # TODO: Learn two different models from the training data
    # using L2 and L infinity losses

        modell2 = minimizeL2(Xfh,yfh)
        modellinf = minimizeLinf(Xfh,yfh)

    # Done: Evaluate the two models' performance (for each model,
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

    # TODO: compute the average losses over runs

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


    # Done: return a 2-by-2 training loss variable and a 2-by-2 test loss variable

    return avg_train_loss, avg_test_loss
        

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#def logisticRegObj(w, X, y):
#    n, d = X.shape
#    #return np.linalg.inv(X.T @ X) @ (X.T @ y)
#    z = (1/n) * (-y.T @ np.log(sigmoid(X@w)) - (1-y).T @ np.log(1-sigmoid(X@w)))
#    return z.item()


def logisticRegObj(w, X, y):
    n, d = X.shape
    z = X @ w
    loss = (1/n) * (np.sum(np.logaddexp(0, z)) - np.sum(y * z))
    return float(loss)



def logisticRegGrad(w, X, y):
    n, d = X.shape
    #(1/n)X.T(sigmoid(Xw)-y)
    z = (1/n)*(X.T @ (sigmoid(X@w)-y))
    return z

    


#2a
def linearRegL2Obj(w, X, y):
    n, _ = X.shape
    obj_value = (1 / (2 * n)) * (((X @ w - y).T) @ (X @ w - y))[0][0]
    return obj_value

def linearRegL2Grad(w, X, y):
    n, _ = X.shape
    obj_gradient = (1/n) * X.T @ (X @ w - y)
    return obj_gradient

def find_opt(obj_func, grad_func, X, y):
    d = X.shape[1]
    w_0 = np.random.rand(d)

    def func(w):
        w = w[:, None]
        obj_value = obj_func(w, X, y)
        return obj_value

    def gd(w):
        w = w[:, None]
        grad = grad_func(w, X, y)[:, 0]  # turn back into 1-d array
        return grad

    return minimize(func, w_0, jac=gd)['x'][:, None]




#q2d

def quad_expand(X):
    return np.concatenate([X, X**2], axis=1)

def preprocessBCW(dataset_folder):
    filepath = os.path.join(dataset_folder,'wdbc.data')
    file = pd.read_csv(filepath, header=None)
    y = file.iloc[:, 1].map({'B': 0.0, 'M': 1.0}).to_numpy().reshape(-1, 1)
    X = file.iloc[:, 2:].to_numpy(dtype=float)
    return X, y

def runBCW(dataset_folder):
    X, y = preprocessBCW(dataset_folder)
    X = quad_expand(X)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 100
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])
    # DONE: Change the following random seed to one of your student IDs
    np.random.seed(101260693)
    for r in range(n_runs):
        # DONE: Randomly partition the dataset into two parts (50%
        # training and 50% test) 
        n = X.shape[0]
        permed = np.random.permutation(n)
        split = n // 2
        idx_fh = permed[:split]
        idx_sh = permed[split:]

        Xfh, yfh= X[idx_fh], y[idx_fh]
        Xsh, ysh = X[idx_sh], y[idx_sh]

        
        
        w = find_opt(logisticRegObj, logisticRegGrad, Xfh, yfh)
        # DONE: Evaluate the model's accuracy on the training
        # data. Save it to `train_acc` //
        yhat_train = (sigmoid(Xfh @ w) >= 0.5).astype(float)
        train_acc[r] = 1.0 - np.mean(yhat_train != yfh)
        
        # DONE: Evaluate the model's accuracy on the test
        # data. Save it to `test_acc`
        yhat_test  = (sigmoid(Xsh @ w) >= 0.5).astype(float)
        test_acc[r]  = 1.0 - np.mean(yhat_test  != ysh)

    # DONE: compute the average accuracies over runs
    avg_train_acc = float(np.mean(train_acc))
    avg_test_acc = float(np.mean(test_acc))
    # DONE: return two variables: the average training accuracy and average test accuracy
    return avg_train_acc, avg_test_acc


        

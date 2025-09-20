
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

def hello_world():
    print("Hello world")

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
    # print("Printing c:")
    # print(c, c.shape, "\n")

    # Technology matrix
    G_1 = np.concatenate([np.zeros([1, d]), -np.ones([1,1])], axis=1)
    G_2 = np.concatenate([X, -(np.ones([n, 1]))], axis=1)
    G_3 = np.concatenate([-X, -(np.ones([n, 1]))], axis=1)
    G = np.concatenate([G_1, G_2, G_3], axis=0)
    # print("Printing G:")
    # print(G, G.shape, "\n")

    # Right hand side vector
    h = np.concatenate([np.zeros((1, 1)), y, -y], axis=0)
    # print("Printing h:")
    # print(h, h.shape, "\n")

    # Convert to solver matrix
    c_final = matrix(c)
    G_final = matrix(G)
    h_final = matrix(h)

    # Solve for unknowns
    solvers.options['show_progress'] = False
    sol = solvers.lp(c_final, G_final, h_final)

    # Return only values for w and do not include threshold \delta
    final_sol = np.array(sol['x'])[:-1, :]
    # print(np.array(sol['x']))
    # print(final_sol)
    return final_sol

'''
    synRegExperiments
    Output: A 2x2 matrix of average traning losses and a 2x2 matrix of average test losses for L2 & Linf models and L2 & Linf losses
'''
def synRegExperiments():
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
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # DONE: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the training
        # data). Save them to `train_loss`

        # TRAINING DATA
            # L2 model
        L2model_L2_loss = np.mean(((Xtrain @ w_L2) - ytrain) ** 2) 
        L2model_Linf_loss = np.max(np.abs((Xtrain @ w_L2) - ytrain))
        train_loss[r][0][0] = L2model_L2_loss
        train_loss[r][0][1] = L2model_Linf_loss
            # L_inf model
        Linfmodel_L2_loss = np.mean(((Xtrain @ w_Linf) - ytrain) ** 2) 
        Linfmodel_Linf_loss = np.max(np.abs((Xtrain @ w_Linf) - ytrain))
        train_loss[r][1][0] = Linfmodel_L2_loss
        train_loss[r][1][1] = Linfmodel_Linf_loss

        # DONE: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the test
        # data). Save them to `test_loss`
        # TEST DATA
            # L2 model
        L2model_L2_loss = np.mean(((Xtest @ w_L2) - ytest) ** 2) 
        L2model_Linf_loss = np.max(np.abs((Xtest @ w_L2) - ytest))
        test_loss[r][0][0] = L2model_L2_loss
        test_loss[r][0][1] = L2model_Linf_loss
            # L_inf model
        Linfmodel_L2_loss = np.mean(((Xtest@ w_Linf) - ytest) ** 2) 
        Linfmodel_Linf_loss = np.max(np.abs((Xtest @ w_Linf) - ytest))
        test_loss[r][1][0] = Linfmodel_L2_loss
        test_loss[r][1][1] = Linfmodel_Linf_loss

    # DONE: compute the average losses over runs
    # TRAINING DATA - AVG TRAIN LOSS
    avg_train_loss = np.zeros([2,2])

    # Total L2 model - L2 losses and Linf losses
    total_L2model_L2_loss = np.sum(train_loss[:, 0, 0]) 
    total_L2model_Linf_loss = np.sum(train_loss[:, 0, 1]) 

    # Total Linf model - L2 losses and Linf losses
    total_Linfmodel_L2_loss = np.sum(train_loss[:, 1, 0]) 
    total_Linfmodel_Linf_loss = np.sum(train_loss[:, 1, 1]) 
    
    avg_train_loss[0][0] = total_L2model_L2_loss / n_runs     # Avg L2 model L2 loss
    avg_train_loss[0][1] = total_L2model_Linf_loss / n_runs   # Avg L2 model Linf loss 
    avg_train_loss[1][0] = total_Linfmodel_L2_loss / n_runs   # Avg Linf model L2 loss
    avg_train_loss[1][1] = total_Linfmodel_Linf_loss / n_runs # Avg Linf model Linf loss
    print(avg_train_loss)
    
    # TEST DATA - AVG TEST LOSS
    avg_test_loss = np.zeros([2,2])
    
    # Total L2model L2 losses and Linf losses
    total_L2model_L2_loss = np.sum(test_loss[:, 0, 0]) 
    total_L2model_Linf_loss = np.sum(test_loss[:, 0, 1]) 

    # Total Linf Model L2 losses and Linf losses
    total_Linfmodel_L2_loss = np.sum(test_loss[:, 1, 0]) 
    total_Linfmodel_Linf_loss = np.sum(test_loss[:, 1, 1]) 
    
    avg_test_loss[0][0] = total_L2model_L2_loss / n_runs     # Avg L2 model L2 loss
    avg_test_loss[0][1] = total_L2model_Linf_loss / n_runs   # Avg L2 model Linf loss 
    avg_test_loss[1][0] = total_Linfmodel_L2_loss / n_runs   # Avg Linf model L2 loss
    avg_test_loss[1][1] = total_Linfmodel_Linf_loss / n_runs # Avg Linf model Linf loss
    print(avg_test_loss)
    
    # DONE: return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return avg_train_loss, avg_test_loss


if __name__ == "__main__":
    # X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
    # y = np.array([1, 2, 3])
    # minimizeLinf(X, y)
    synRegExperiments()

def preprocessCCS(dataset_folder):
    filepath = os.path.join(dataset_folder,"Concrete_Data.xls")
    file = pd.read_excel(filepath,skiprows=1)
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

    # TODO: Evaluate the two models' performance (for each model,
    # calculate the L2 and L infinity losses on the training
    # data). Save them to `train_loss`



    # TODO: Evaluate the two models' performance (for each model,
    # calculate the L2 and L infinity losses on the test
    # data). Save them to `test_loss`
    # TODO: compute the average losses over runs
    # TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable
        


    

    
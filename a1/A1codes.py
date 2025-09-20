
import numpy as np
from cvxopt import matrix, solvers

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
    Input loss: the 2x2 loss matrix
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
    synRegExperiments
    Output: A 2x2 matrix of average traning losses and a 2x2 matrix of average test losses for L2 & Linf models and L2 & Linf losses
'''
def synRegExperiments():
    def genData(n_points, is_training=False):
        '''
        This function generate synthetic data
        '''
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

        # X = np.random.randn(n_points, d) # input matrix
        # X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        # y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        # if is_training:
        #     y[0] *= -0.1

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
        print(w_L2)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # TRAINING DATA: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the training
        # data). Save them to `train_loss`
            # L2 model
        L2model_L2_loss = L2_loss(Xtrain, w_L2, ytrain) 
        L2model_Linf_loss = Linf_loss(Xtrain, w_L2, ytrain)
        train_loss[r][0][0] = L2model_L2_loss
        train_loss[r][0][1] = L2model_Linf_loss
            # L_inf model
        Linfmodel_L2_loss = L2_loss(Xtrain, w_Linf, ytrain)
        Linfmodel_Linf_loss = Linf_loss(Xtrain, w_Linf, ytrain) 
        train_loss[r][1][0] = Linfmodel_L2_loss
        train_loss[r][1][1] = Linfmodel_Linf_loss

        # TEST DATA: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the test
        # data). Save them to `test_loss`
            # L2 model
        L2model_L2_loss = L2_loss(Xtest, w_L2, ytest)
        L2model_Linf_loss = Linf_loss(Xtest, w_L2, ytest)
        test_loss[r][0][0] = L2model_L2_loss
        test_loss[r][0][1] = L2model_Linf_loss
            # L_inf model
        Linfmodel_L2_loss = L2_loss(Xtest, w_Linf, ytest)
        Linfmodel_Linf_loss = Linf_loss(Xtest, w_Linf, ytest)
        test_loss[r][1][0] = Linfmodel_L2_loss
        test_loss[r][1][1] = Linfmodel_Linf_loss

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
    print(avg_train_loss)
    
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
    print(avg_test_loss)
    
    # Return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return avg_train_loss, avg_test_loss


if __name__ == "__main__":
    synRegExperiments()

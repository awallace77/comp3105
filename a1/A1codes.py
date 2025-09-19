
import numpy as np
from cvxopt import matrix, solvers

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
    # TODO: Andrew to implement from result in question 1 b.5
    '''
    cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])
    '''
    # nxd dimensions
    n, d = X.shape
    c = np.zeros([d, 1])
    c[d - 1][0] = 1

    G_1 = np.array([[0 for i in range(d + 1)]])
    G_2 = np.concatenate([X, (np.array([[-1 for i in range(n)]]).T)], axis=1)
    G_3 = np.concatenate([X, (np.array([[1 for i in range(n)]]).T)], axis=1)

    G = np.concatenate([G_1, G_2, G_3], axis=0)

    # u = [w \delta]^T
    # h = [0, y_1, ..., y_n, -y_1, ... -y_n]
    
    print(G)

if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    minimizeLinf(X, y)
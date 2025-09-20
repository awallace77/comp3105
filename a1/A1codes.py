
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
    '''
    cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])
    '''
    
    # nxd dimensions
    n, d = X.shape

    # Objective coefficients
    c = np.concatenate([np.zeros([d, 1]), np.ones([1,1])])
    print("Printing c:")
    print(c, c.shape, "\n")

    # Technology matrix
    G_1 = np.concatenate([np.zeros([1, d]), -np.ones([1,1])], axis=1)
    G_2 = np.concatenate([X, -(np.ones([n, 1]))], axis=1)
    G_3 = np.concatenate([-X, -(np.ones([n, 1]))], axis=1)
    G = np.concatenate([G_1, G_2, G_3], axis=0)
    print("Printing G:")
    print(G, G.shape, "\n")

    # Right hand side vector
    h = np.concatenate([np.zeros((1, 1)), y, -y], axis=0)
    print("Printing h:")
    print(h, h.shape, "\n")

    # Convert to solver matrix
    c_final = matrix(c)
    G_final = matrix(G)
    h_final = matrix(h)

    # Solve for unknowns
    sol = solvers.lp(c_final, G_final, h_final)
    final_sol = np.array(sol['x'])[:-1, :]
    print(np.array(sol['x']))
    print(final_sol)
    return final_sol


# if __name__ == "__main__":
    # X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
    # y = np.array([1, 2, 3])
    # minimizeLinf(X, y)
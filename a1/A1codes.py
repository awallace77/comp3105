
import numpy as np

def hello_world():
    print("Hello world")


# minimizeL2
# Input X: a nxd matrix 
# Input y: a nx1 target/label vector
# Output: dx1 vector of parameters w corresponding to the solution of the L2 losses
# w = (X.T X)^{-1} X.T y
def minimizeL2(X, y):
    return np.linalg.inv((np.transpose(X) @ X)) @ (np.transpose(X) @ y)
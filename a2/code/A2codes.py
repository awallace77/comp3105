import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers

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
        Calculates the optimal parameters and scalar intercept to the solution of the regularized ExpLinear Loss
        Parameters:
            X: nxd input matrix
            y: nx1 target vector
            lamb: regularization hyper-parameter 
    """

    _, d = X.shape
    w0 = np.ones(d + 1)

    def expLinear(w):
        # margin
        m = y * (X @ w[1:] + w[0]) 
        losses = np.where(-m >= 0, 1 - m, np.exp(-m))
        reg = (lamb / 2) * np.sum(w[1:].T @ w[1:])
        sum = np.sum(losses) + reg
        return sum
    
    res = minimize(expLinear, w0)
    w = res.x[1:]
    w0 = res.x[0]

    # TODO: REMOVE AFTER DEBUG
    print("Optimization success:", res.success)
    print("Final loss:", res.fun)
    print("Optimal weights:", res.x)
    pred = np.sign(X_complex @ res.x[1:] + res.x[0])
    accuracy = np.mean(pred == y_complex)
    print(f"Accuracy: {accuracy*100:.2f}%")

    return w, w0


# Question 1(b)
def minHinge(X, y, lamb, stabilizer=1e-5):
    """
        Calculates the parameters and scalar intercept of the solution to the regularized hinge loss
    """
    n, d = X.shape
    
    q = np.concatenate([np.zeros(d + 1), np.ones(n)])
    G1 = np.hstack([np.zeros((n, d)), np.zeros((n, 1)), -np.eye(n)])
    G2 = np.hstack([-np.diag(y) @ X, -y[:, None], -np.eye(n)])
    G = np.vstack([G1, G2])
    h = np.concatenate([np.zeros((n, 1)), -np.ones((n, 1))])

    P = np.block([
        [lamb * np.eye(d), np.zeros((d, 1)), np.zeros((d, n))],
        [np.zeros((1, d)), np.zeros((1, 1)), np.zeros((1, n))],
        [np.zeros((n, d)), np.zeros((n, 1)), np.zeros((n, n))],
    ])

    # Add stabilizer
    P = P + stabilizer * np.eye(n + d + 1)

    # Convert to cvx matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    sol = solvers.qp(P, q, G, h)
    u = np.array(sol['x']).flatten()
    w = u[:d]
    w0 = u[d] 
    return  w, w0




if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility

    # QUESTION 1A
    n_samples = 200

    # Class +1: centered at (2, 2)
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])

    # Class -1: centered at (-2, -2)
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

    # Combine
    X_complex = np.vstack([X_pos, X_neg])
    y_complex = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

    # Add some noise / overlap to make it less trivial
    X_complex[:20] += np.random.randn(20, 2) * 4

    lamb = 0.1
    w, w0 = minExpLinear(X_complex, y_complex, lamb)
    


    # QUESTION 1B
    # --- Test Dataset ---
    # Simple 2D linearly separable data
    X = np.array([
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 1]
    ])

    y = np.array([1, 1, -1, -1])  # labels

    # Regularization parameter
    lamb = 1.0

    # Solve SVM
    w, w0 = minHinge(X, y, lamb)

    print("Weights:", w)
    print("Bias:", w0)

    # --- Optional: Test predictions ---
    def predict(X, w, w0):
        return np.sign(X @ w + w0)

    preds = predict(X, w, w0)
    print("Predictions:", preds)
    print("True labels:", y)



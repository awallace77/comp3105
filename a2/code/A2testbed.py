# COMP 3105 Assignment 2
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
import A2codes as A2codes
from A2helpers import plotModel, plotAdjModel, plotDualModel, polyKernel, generateData, linearKernel, gaussKernel
import time

def _plotCls():

	n = 100
	lamb = 0.01

	gen_model = 3
	kernel_func = lambda X1, X2: gaussKernel(X1, X2, 1)


	# Generate data
	Xtrain, ytrain = generateData(n=n, gen_model=gen_model)

	# Learn and plot results
	# Primal
	w, w0 = A2codes.minExpLinear(Xtrain, ytrain, lamb)
	plotModel(Xtrain, ytrain, w, w0, A2codes.classify, "minExpLinear")

	w, w0 = A2codes.minHinge(Xtrain, ytrain, lamb)
	plotModel(Xtrain, ytrain, w, w0, A2codes.classify, "minHinge")
	
	# Adjoint
	a, a0 = A2codes.adjExpLinear(Xtrain, ytrain, lamb, kernel_func)
	plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, A2codes.adjClassify, "adjExpLinear")

	a, a0 = A2codes.adjHinge(Xtrain, ytrain, lamb, kernel_func)
	plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, A2codes.adjClassify, "adjHinge")

	# Dual
	a, b = A2codes.dualHinge(Xtrain, ytrain, lamb, kernel_func)
	plotDualModel(Xtrain, ytrain, a, b, lamb, kernel_func, A2codes.dualClassify, "dualHinge")


if __name__ == "__main__":

	'''
	_plotCls()
	# Question 1 (d)
	avg_train_acc, avg_test_acc = synExperimentsRegularize()

	# Question 2 (a)
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

	# Question 2 (b)
	# Checking correctness by comparing to q1 (b) 
	w, w0 = minHinge(Xtrain, ytrain, lamb)
	a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)

	hinge = Xtrain @ w + w0
	adj_hinge = K @ a + a0

	print(f"HINGE:\n {hinge}")
	print(f"ADJHINGE:\n {adj_hinge}")

	# Question 2 (d)
	avg_train_acc, avg_test_acc = synExperimentsKernel()
	print(f"Average TRAIN accuracy:\n{avg_train_acc}")
	print(f"Average TEST accuracy:\n{avg_test_acc}")

	'''

	# Synthetic Experiments
	start = time.time()
	avg_train_acc, avg_test_acc = A2codes.synExperimentsRegularize()
	end = time.time()
	print(f"Average TRAIN accuracy:\n{avg_train_acc}")
	print(f"Average TEST accuracy:\n{avg_test_acc}")
	print(f"synExperimentsRegularize took {end - start:.4f} seconds. Should be approx 20s")

	start = time.time()
	avg_train_acc, avg_test_acc = A2codes.synExperimentsKernel()
	end = time.time()
	print(f"Average TRAIN accuracy:\n{avg_train_acc}")
	print(f"Average TEST accuracy:\n{avg_test_acc}")
	print(f"synExperimentsKernel took {end - start:.4f} seconds. Should be approx 5 minutes")

	start = time.time()
	cv_acc, best_lamb, best_kernel = A2codes.cvMnist(
		"/home/andrew/cu/comp3105/a2/code",
		[0.001, 0.01, 0.1, 1.],
		[
			linearKernel,
			lambda X1, X2: polyKernel(X1, X2, 2),
			lambda X1, X2: gaussKernel(X1, X2, 1.0)
		],
		k=5
	)
	end = time.time()
	print(f"cv_acc:\n{cv_acc}")
	print(f"best_lamb:\n{best_lamb}")
	print(f"best_kernel:\n{best_kernel}")
	print(f"cvMnist took {end - start:.4f} seconds. Should be around 30s")


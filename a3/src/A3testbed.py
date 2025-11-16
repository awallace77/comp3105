# COMP 3105 Assignment 3
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
from matplotlib import pyplot as plt
import numpy as np
import A3codes as A3codes
from A3helpers import augmentX, gaussKernel, plotModel, generateData, plotPoints, synClsExperiments


def _plotCls():

	n = 100
	model = 2

	# Generate data
	Xtrain, Ytrain = generateData(n=n, gen_model=model)
	Xtrain = augmentX(Xtrain)

	# Learn and plot results
	W = A3codes.minMulDev(Xtrain, Ytrain)
	print(f"Train accuaracy {A3codes.calculateAcc(Ytrain, A3codes.classify(Xtrain, W))}")

	plotModel(Xtrain, Ytrain, W, A3codes.classify, f"cls_train{model}")

	return

def _testCls():	
	train_acc, test_acc = synClsExperiments(minMulDev=A3codes.minMulDev, classify=A3codes.classify, calculateAcc=A3codes.calculateAcc)
	print("Train accuracy: \n", train_acc)
	print("Test accuracy: \n", test_acc)



def _testPCA():
	train_acc, test_acc = A3codes.synClsExperimentsPCA()
	print("Train accuracy: \n", train_acc)
	print("Test accuracy: \n", test_acc)
	return


def _plotKmeans():

	n = 100
	k = 3

	Xtrain, _ = generateData(n, gen_model=2)

	Y, U, obj_val = A3codes.kmeans(Xtrain, k)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.savefig(f"plotKmeans.png")
	plt.clf()  # clears the figure for the next plot
	# plt.show()

	return


def _plotKernelKmeans():
	Xtrain, _ = generateData(n=100, gen_model=3)
	kernel_func = lambda X1, X2: gaussKernel(X1, X2, 0.25)

	n_points = 100
	n_clusters = 2
	

	# init_Y = np.random.rand(100, 2)
	def repeatKmeans(n_runs=100):
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
		for r in range(n_runs):
			init_Y = np.zeros((n_points, n_clusters))
			for i in range(n_points):
				init_Y[i, np.random.randint(n_clusters)] = 1

			Y, obj_val = A3codes.kernelKmeans(Xtrain, kernel_func, 2, init_Y)
			#if obj_val is smallest then keep that one
			if obj_val<best_obj_val:
				best_obj_val = obj_val
				best_Y = Y
		#return Y and U of smallest Object Value
		return best_Y, best_obj_val

	# Y, obj_val = A3codes.kernelKmeans(Xtrain, kernel_func, 2, init_Y)
	Y, obj_val = repeatKmeans() 
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.savefig(f"plotKernelKmeans.png")
	plt.clf()  # clears the figure for the next plot
	# plt.show()
	return


if __name__ == "__main__":

	_plotCls()
	_testCls()
	_testPCA()
	_plotKmeans()
	_plotKernelKmeans()

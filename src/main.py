import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import csv

# Parameters: theta - Vector of coefficients
#             x     - 1-D Vector of a specific training example
# Vectors must be of the same length
# x[*][0] = 1
def hyp(x, theta):
	z = 0
	for i in range (0, len(theta)):
		z += theta[i] * x[i]
	return 1 / ( 1 + math.exp(-z))


# Reprents the "error" in the presumed function. This is what is minimized (line error)
# Parameters: x - 2-D Vector representing the training data
#             y - 1-D Vector of real numbers representing results
#			  theta - 1-D Vector of coefficients
def cost(x, y, theta):
	summation = 0
	i = 0
	for i in range(len(x)):
		summation += y[i]*math.log(hyp(x[i],theta),10) + (1-y[i])*math.log(1-hyp(x[i],theta),10)
	return -summation/len(x)

# Parameters: x - 2-D Vector of training data
#             y - 1-D Vector of real numbers representing results
#			  theta - 1-D Vector of coefficients
# 			  alpha - Learning rate	meters: 
def step(x, y, theta, alpha):
	total = 0
	temp_theta = []
	for j in range(0, len(x[0])):
		for i in range(0, len(x)):
			total += x[i][j] * (hyp(x[i], theta) - y[i])
		temp_theta.append(theta[j] - alpha*total/float(len(x)))
		total = 0
	return temp_theta

# Parameters: x - 2-D Vector of training data
#             y - 1-D Vector of real numbers representing results
#			  theta - 1-D Vector of coefficients
# 			  max_itt - Spacify the number of times "theta" is updated
# 			  alpha - Learning rate	
def gradient_descent_runner(x, y, theta, max_itt, alpha):
	last_cost = cost(x, y, theta)
	count = 0
	for i in range(max_itt):
		theta = step(x, y, theta, alpha)
		curr_cost = cost(x, y, theta)
		if last_cost < curr_cost:
			break
		last_cost = curr_cost
		count += 1
	print "Number of itterations (debugging): ", count
	return theta

# Reads in float's from the CSV
def float_wrapper(reader):
    for v in reader:
        yield map(float, v)

# Parameters: file_name - The name of the file that will populate our data
def populate(file_name):
    x = []
    y = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/' + file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        reader = float_wrapper(reader)
        for row in reader:
            # This is for completeness, first factor is always 1
            x.append([1])
            x[len(x) - 1] += row[0:len(row) - 1]
            y.append(row[len(row) - 1])
    return (x, y)

# Formatted for 1 feature right now
# x[*][0] = 1
def plot(x, y, theta):
	print "Plotting..."
	# Training data
	plt.plot(x,y,'ro')
	# Optimized Function
	formula = ""
	for i in range (0, len(theta)):
		if i == 0:
			formula += str(theta[0])
		else:
			formula += "+" + str(theta[i]) + "*x_range"
	x_range = np.linspace(-1.0, 1.5, num=20)
	formula = "1 / ( 1 + (math.e)**(" + formula + "))"
	y_predict = eval(formula)
	plt.plot(x_range, y_predict)
	plt.axis([-0.5,1.5,-0.1,1.1])
	plt.show()

# Notes:
# 	- Must have at least 1 training set
def run():
	x, y = populate('grain_size.csv')
	theta = [0] * len(x[0])

	alpha = 0.001
	max_itt = 10000

	print 'Running...'
	theta = gradient_descent_runner(x, y, theta, max_itt, alpha)

	plot(x, y, theta)
	print "Final theta: ", theta

if __name__ == '__main__':
    run()


######################################################################################################
# Notes
# In logistic regression the Y {0,1}
# The reason for the different cost function than linear regression is becasue
#	the new cost function guarantees a convex shape, 1 local min is necessary
#	for gradient descent to work properly
# The differance between linear and logistic is the hypothesis. The gradient descent is the same
# Feature scaling exists for logistic regression as well
# Logistic Regression can be applied to nominal values (non numberic) which is an advantage to linear
# 	The nominal value can also be y (the result)


## Python Notes:
# np.array is good for eval, doesn't have commas as delimeters
# np.linspace(start, stop, num=num_divisions)

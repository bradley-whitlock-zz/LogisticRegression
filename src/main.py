# This is the main runner
import math

# Parameters: theta - Vector of coefficients
#             x     - 1-D Vector of a specific training example
def hyp(x, theta):
	z = 0
	for t in theta:
		z += theta[t] * x[t]
	return 1 / ( 1 + math.exp(-z))


# Reprents the "error" in the presumed function. This is what is minimized (line error)
# Parameters: x - 2-D Vector representing the training data
#             y - 1-D Vector of real numbers representing results
#			  t - 1-D Vector of coefficients
def cost(x, y, t):
	summation = 0
	for i in range(x):
		summation += y[i]*math.log(hyp(x[i],t),10) + (1-y[i])*math.log(1-hyp(x[i],t),10)
	return -summation/len(x)

# Parameters: x - 2-D Vector of training data
#             y - 1-D Vector of real numbers representing results
#			  t - 1-D Vector of coefficients
# 			  alpha - Learning rate	meters: 
def step(x, y, theta, alpha):
	print "hello"

# Parameters: x - 2-D Vector of training data
#             y - 1-D Vector of real numbers representing results
#			  t - 1-D Vector of coefficients
# 			  max_itt - Spacify the number of times "theta" is updated
# 			  alpha - Learning rate	
def gradient_descent_runner(x, y, theta, max_itt, alpha):
	last_cost = cost(x, y, theta)
	for i in range(max_itt):
		theta = step(x, y, theta, alpha)


def run():
	print "Wow Logistic Regression is Easy and Fun"


if __name__ == '__main__':
    run()
#!/usr/bin/python2.7
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    # Get the dimension of input data
    d = np.size(data_in,1)-1
    # Initialize the weight vector
    w = np.zeros(d)
    # Iterate to update the weight vector
    iterations = 0
    while 1:
        w_update = 0
        for i in range(np.size(data_in,0)):
            if np.sign(data_in[i,:d].dot(w)) != data_in[i,-1]:
                w += data_in[i,-1]*data_in[i,:d]
                w_update = 1
                iterations += 1
                break
        if w_update == 0:
            break
    
    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW0
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # Your code here, assign the values to num_iters and bounds_minus_ni:
    for repeat in range(num_exp):
        # Generate a random optimal seperator w_star
        w_star = np.random.uniform(0,1,(d+1,))
        w_star[0] = 0
        
        # Generate a random training set
        X_train = np.random.uniform(-1,1,(N,d+1))
        X_train[:,0] = 1
        y_train = np.sign(X_train.dot(w_star))
        
        # Calculate the theoretical bound
        R = max(np.linalg.norm(X_train, ord=2, axis=1))
        rho = min(y_train*(X_train.dot(w_star)))
        w_norm = np.linalg.norm(w_star, ord=2)
        bound = R**2*w_norm**2/rho**2
        
        # Do the perceptron experiment
        w, num_iters[repeat] = perceptron_learn(np.insert(X_train, d+1, 
                                                          values=y_train, 
                                                          axis=1))
        bounds_minus_ni[repeat] = bound-num_iters[repeat]
    
    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()

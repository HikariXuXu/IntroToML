#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    
    N = X.shape[0] # The number of data
    
    # Merge X to get X_merge with an initial column of 1s
    X1 = np.ones(N)
    X_merge = np.c_[X1, X]
    
    # Prediction
    y_pred = 1/(np.ones(N)+np.exp(-X_merge.dot(w)))
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = -1
    
    binary_error = sum(y_pred!=y)/N

    return binary_error


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:
    
    N = X.shape[0] # The number of data
    w = w_init
    
    # Merge X to get X_merge with an initial column of 1s
    X1 = np.ones(N)
    X_merge = np.c_[X1, X]
    
    # Training
    for t in range(int(max_its)):
        grad = -1/N*np.sum(np.array([(y/(np.ones(N)+np.exp(X_merge.dot(w)*y)))]).repeat([X.shape[1]+1], axis=0).T*X_merge, axis=0)
        w = w - eta*grad
        if sum(abs(grad) < grad_threshold) == len(grad):
            break
    
    # Calculate the in-sample error
    e_in = 1/N*sum(np.log(np.ones(N)+np.exp(-X_merge.dot(w)*y)))
    
    return t+1, w, e_in


def main():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')
    
    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')
    
    # Your code here
    
    # Divide X and y from the original data
    X_train = np.array(train_data.drop(['heartdisease::category|0|1'], axis=1))
    y_train = np.array(train_data['heartdisease::category|0|1'])
    X_test = np.array(test_data.drop(['heartdisease::category|0|1'], axis=1))
    y_test = np.array(test_data['heartdisease::category|0|1'])
    
    # Transform the value 0 in y into -1
    y_train[y_train==0] = -1
    y_test[y_test==0] = -1
    
    # Set up the parameters
    eta = 1e-5
    grad_threshold = 1e-3
    max_its = [1e4, 1e5, 1e6]
    
    num_of_iters = np.zeros(3)
    w_trained = np.zeros((3, X_train.shape[1]+1))
    e_in_results = np.zeros(3)
    training_time = np.zeros(3)
    training_error = np.zeros(3)
    testing_error = np.zeros(3)
    
    for i in range(3):
        # Initial the weight vector
        w_init = np.zeros(X_train.shape[1]+1)
        
        time_start = time.time()
        
        # Train the model
        num_of_iters[i], w_trained[i], e_in_results[i] = logistic_reg(X_train, 
                                                                      y_train, 
                                                                      w_init, 
                                                                      max_its[i], 
                                                                      eta, 
                                                                      grad_threshold)
        
        time_end = time.time()
        training_time[i] = time_end - time_start
        
        # Output the binary error
        training_error[i] = find_binary_error(w_trained[i], X_train, y_train)
        testing_error[i] = find_binary_error(w_trained[i], X_test, y_test)
        
        # Output
        print('------------------------------')
        print('max_its =',max_its[i])
        print('E_in =', e_in_results[i])
        print('training_error =', training_error[i])
        print('testing_error =', testing_error[i])
        print('training_time =', training_time[i])
    
    # Normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_norm = (X_train-np.array([mean]).repeat(X_train.shape[0], axis=0))/std
    X_test_norm = (X_test-np.array([mean]).repeat(X_test.shape[0], axis=0))/std
    
    # Set up the parameters
    eta_list = [0.01, 0.1, 1, 4, 7, 7.5, 7.6, 7.7]
    max_iterations = 1e6
    new_grad_threshold = 1e-6
    
    cross_entropy = np.zeros(len(eta_list))
    training_binary_error = np.zeros(len(eta_list))
    testing_binary_error = np.zeros(len(eta_list))
    iterations = np.zeros(len(eta_list))
    training_process_time = np.zeros(len(eta_list))
    
    for j in range(len(eta_list)):
        # Initial the weight vector
        w_init = np.zeros(X_train.shape[1]+1)
        
        time_start = time.time()
        
        # Train the model
        iterations[j], w_out, cross_entropy[j] = logistic_reg(X_train_norm, 
                                                              y_train, 
                                                              w_init, 
                                                              max_iterations, 
                                                              eta_list[j], 
                                                              new_grad_threshold)
        
        time_end = time.time()
        training_process_time[j] = time_end - time_start
        
        # Output the binary error
        training_binary_error[j] = find_binary_error(w_out, X_train_norm, y_train)
        testing_binary_error[j] = find_binary_error(w_out, X_test_norm, y_test)
        
        # Output
        print('------------------------------')
        print('eta =',eta_list[j])
        print('E_in =', cross_entropy[j])
        print('training_error =', training_binary_error[j])
        print('testing_error =', testing_binary_error[j])
        print('number_of_iterations =', iterations[j])
        print('training_time =', training_process_time[j])


if __name__ == "__main__":
    main()

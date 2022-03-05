#!/usr/bin/python3
# Homework 3 Code
import numpy as np
from sklearn import preprocessing

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

def l1_logistic_reg(X, y, w_init, max_its, lbd, eta, grad_threshold):
    # l1_logistic_reg learn logistic regression model with L1 regularizer using truncated gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        lbd: amount of regularization
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        w : weight vector

    # Your code here, assign the proper values to t, w, and e_in:
    
    N = X.shape[0] # The number of data
    w = w_init
    
    # Merge X to get X_merge with an initial column of 1s
    X1 = np.ones(N)
    X_merge = np.c_[X1, X]
    
    # Training
    for t in range(int(max_its)):
        grad = -1/N*np.sum(np.array([(y/(np.ones(N)+np.exp(X_merge.dot(w)*y)))]).repeat([X.shape[1]+1], axis=0).T*X_merge, axis=0)
        w1 = w - eta*grad
        w = w1 - eta*lbd*np.sign(w)
        w[(np.sign(w)-np.sign(w1)!=0) * (w1!=0) == 1] = 0
        if sum(abs(grad) < grad_threshold) == len(grad):
            break
    
    return w

def l2_logistic_reg(X, y, w_init, max_its, lbd, eta, grad_threshold):
    # l2_logistic_reg learn logistic regression model with L2 regularizer using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        lbd: amount of regularization
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        w : weight vector

    # Your code here, assign the proper values to t, w, and e_in:
    
    N = X.shape[0] # The number of data
    w = w_init
    
    # Merge X to get X_merge with an initial column of 1s
    X1 = np.ones(N)
    X_merge = np.c_[X1, X]
    
    # Training
    for t in range(int(max_its)):
        grad = -1/N*np.sum(np.array([(y/(np.ones(N)+np.exp(X_merge.dot(w)*y)))]).repeat([X.shape[1]+1], axis=0).T*X_merge, axis=0)
        w = (1-2*eta*lbd)*w - eta*grad
        if sum(abs(grad) < grad_threshold) == len(grad):
            break
    
    return w


#def main():
# Load data
X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)

# Transform the value 0 in y into -1
y_train[y_train==0] = -1
y_test[y_test==0] = -1

# Normalization
stdscaler = preprocessing.StandardScaler()
X_train_norm = stdscaler.fit_transform(X_train)
X_test_norm = stdscaler.transform(X_test)

# Set up the parameters
lbd_list = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
eta = 0.01
max_iterations = 1e4
grad_threshold = 1e-6

w_l1_trained = np.zeros((len(lbd_list), X_train.shape[1]+1))
l1_num_of_zeros = np.zeros(len(lbd_list))
l1_testing_error = np.zeros(len(lbd_list))
w_l2_trained = np.zeros((len(lbd_list), X_train.shape[1]+1))
l2_num_of_zeros = np.zeros(len(lbd_list))
l2_testing_error = np.zeros(len(lbd_list))

for i in range(len(lbd_list)):
    # Initial the weight vector
    w_init = np.zeros(X_train.shape[1]+1)
    # Train the Logistic Regression model with L1 regularizer
    w_l1_trained[i] = l1_logistic_reg(X_train_norm, y_train, w_init, max_iterations, lbd_list[i], eta, grad_threshold)
    # Train the Logistic Regression model with L2 regularizer
    w_l2_trained[i] = l2_logistic_reg(X_train_norm, y_train, w_init, max_iterations, lbd_list[i], eta, grad_threshold)
    
    # Calculate the num of zeros in weight vector
    l1_num_of_zeros[i] = sum(w_l1_trained[i]==0)
    l2_num_of_zeros[i] = sum(w_l2_trained[i]==0)
    
    # Testing the trained model
    l1_testing_error[i] = find_binary_error(w_l1_trained[i], X_test_norm, y_test)
    l2_testing_error[i] = find_binary_error(w_l2_trained[i], X_test_norm, y_test)
    
    # Output
    print('------------------------------')
    print('lambda =',lbd_list[i])
    print('L1 testing error =', l1_testing_error[i])
    print('L2 testing error =', l2_testing_error[i])
    print('L1 num of zeros =', l1_num_of_zeros[i])
    print('L2 num of zeros =', l2_num_of_zeros[i])


'''
if __name__ == "__main__":
    main()
'''
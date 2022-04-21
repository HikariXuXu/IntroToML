#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use
    N_train = len(X_train)
    N_test = len(X_test)
    d_t = np.ones(N_train)/N_train
    t = 0
    alpha = []
    y_train_predict = []
    y_test_predict = []
    while t < n_trees:
        clf = DecisionTreeClassifier(max_depth=1, criterion='entropy')
        clf.fit(X_train, y_train, d_t)
        y_train_predict.append(clf.predict(X_train))
        y_test_predict.append(clf.predict(X_test))
        epsilon_t = sum((y_train_predict[-1]!=y_train)*d_t)
        alpha.append(np.log((1-epsilon_t)/epsilon_t)/2)
        gamma_t = np.sqrt((1-epsilon_t)/epsilon_t)
        z_t = gamma_t*epsilon_t+(1-epsilon_t)/gamma_t
        d_t = d_t*np.exp(-alpha[-1]*y_train*y_train_predict[-1])/z_t
        t += 1
    
    y_train_sum = np.zeros(N_train)
    y_test_sum = np.zeros(N_test)
    train_error = []
    test_error = []
    for num in range(1,n_trees+1):
        y_train_sum = y_train_sum + alpha[num-1]*y_train_predict[num-1]
        y_test_sum = y_test_sum + alpha[num-1]*y_test_predict[num-1]
        train_predict = np.sign(y_train_sum)
        test_predict = np.sign(y_test_sum)
        train_error.append(sum(train_predict!=y_train)/N_train)
        test_error.append(sum(test_predict!=y_test)/N_test)

    return train_error, test_error


def extract_data(data, key_values):
    extract_index = [i for i,x in enumerate(data[:,0]) if x in key_values]
    return data[extract_index,:]


#def main_hw5():
# Load data
og_train_data = np.genfromtxt('zip.train')
og_test_data = np.genfromtxt('zip.test')

num_trees = 200

# Extract data whose label = 1 or 3
ext_train_data = extract_data(og_train_data, (1,3))
ext_test_data = extract_data(og_test_data, (1,3))

# Split data
X_train = ext_train_data[:,1:]
y_train = ext_train_data[:,0]
y_train[y_train==1] = 1
y_train[y_train==3] = -1
X_test = ext_test_data[:,1:]
y_test = ext_test_data[:,0]
y_test[y_test==1] = 1
y_test[y_test==3] = -1

train_error_13, test_error_13 = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)

plt.figure(1)
plt.plot(np.arange(1,num_trees+1), train_error_13)
plt.plot(np.arange(1,num_trees+1), test_error_13)
plt.title('1 versus 3')
plt.legend(['train_error', 'test_error'])
plt.xlabel('number of trees')
plt.ylabel('error')
plt.show()

# Extract data whose label = 3 or 5
ext_train_data = extract_data(og_train_data, (3,5))
ext_test_data = extract_data(og_test_data, (3,5))

# Split data
X_train = ext_train_data[:,1:]
y_train = ext_train_data[:,0]
y_train[y_train==3] = 1
y_train[y_train==5] = -1
X_test = ext_test_data[:,1:]
y_test = ext_test_data[:,0]
y_test[y_test==3] = 1
y_test[y_test==5] = -1

train_error_35, test_error_35 = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)

plt.figure(2)
plt.plot(np.arange(1,num_trees+1), train_error_35)
plt.plot(np.arange(1,num_trees+1), test_error_35)
plt.title('3 versus 5')
plt.legend(['train_error', 'test_error'])
plt.xlabel('number of trees')
plt.ylabel('error')
plt.show()


'''
if __name__ == "__main__":
    main_hw5()
'''
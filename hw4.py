#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt


def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function
    
    clf = DecisionTreeClassifier(criterion='entropy')
    N_train = len(y_train)
    N_test = len(y_test)
    # Create bags
    bags_index = np.random.randint(0, N_train, ((num_bags,N_train)))
    # Train each classifier
    clf_list = []
    for i in range(num_bags):
        clf = DecisionTreeClassifier(criterion='entropy')
        clf_list.append(clf.fit(X_train[bags_index[i],:], y_train[bags_index[i]]))
    clf_list = np.array(clf_list)
    # Predict each observation
    oob_index = [[i for i,x in enumerate(bags_index) if np.all(x!=j)] for j in range(N_train)]
    
    oob_predict = np.zeros(N_train)
    for i in range(N_train):
        oob_predict[i] = bagging_predict(clf_list[oob_index[i]], X_train[i].reshape(1,-1))

    clf_predict_test = np.zeros((num_bags, N_test))
    for i in range(num_bags):
        clf_predict_test[i] = clf_list[i].predict(X_test)
    bagging_predict_test = mode(clf_predict_test)[0][0]
    
    # Error
    S = sum(oob_predict!=-1)
    out_of_bag_error = 1/S*sum(oob_predict[oob_predict!=-1]!=y_train[oob_predict!=-1])
    test_error = 1/N_test*sum(bagging_predict_test!=y_test)

    return out_of_bag_error, test_error

def bagging_predict(clf_list, X):
    if len(clf_list) == 0:
        return -1
    else:
        predict = []
        for i in range(len(clf_list)):
            predict.append(clf_list[i].predict(X))
        return mode(predict)[0][0]

def single_decision_tree(X_train, y_train, X_test, y_test):
    N_train = len(y_train)
    N_test = len(y_test)
    
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    
    train_error = 1/N_train*sum(train_predict!=y_train)
    test_error = 1/N_test*sum(test_predict!=y_test)
    
    return train_error, test_error

def extract_data(data, key_values):
    extract_index = [i for i,x in enumerate(data[:,0]) if x in key_values]
    return data[extract_index,:]

def main_hw4():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')
    
    num_bags = 200
    
    # Extract data whose label = 1 or 3
    ext_train_data = extract_data(og_train_data, (1,3))
    ext_test_data = extract_data(og_test_data, (1,3))
    
    # Split data
    X_train = ext_train_data[:,1:]
    y_train = ext_train_data[:,0]
    X_test = ext_test_data[:,1:]
    y_test = ext_test_data[:,0]
    
    
    out_of_bag_error_13 = np.zeros(num_bags)
    test_error_13 = np.zeros(num_bags)
    for num in range(1,num_bags+1):
        # Run bagged trees
        out_of_bag_error_13[num-1], test_error_13[num-1] = bagged_trees(X_train, y_train, X_test, y_test, num)
        # Run single tree
        sdt_train_error_13, sdt_test_error_13 = single_decision_tree(X_train, y_train, X_test, y_test)
    
    plt.figure(1)
    plt.plot(np.arange(1,num_bags+1), out_of_bag_error_13)
    plt.legend(['1 versus 3'])
    plt.xlabel('number of bags')
    plt.ylabel('OOB error')
    plt.show()
    
    print('1 versus 3')
    print('Bagging:')
    print('OOB error:', out_of_bag_error_13[num_bags-1])
    print('test error:', test_error_13[num_bags-1])
    print('Single:')
    print('test error:', sdt_test_error_13)
    
    
    # Extract data whose label = 3 or 5
    ext_train_data = extract_data(og_train_data, (3,5))
    ext_test_data = extract_data(og_test_data, (3,5))
    
    # Split data
    X_train = ext_train_data[:,1:]
    y_train = ext_train_data[:,0]
    X_test = ext_test_data[:,1:]
    y_test = ext_test_data[:,0]
    
    out_of_bag_error_35 = np.zeros(num_bags)
    test_error_35 = np.zeros(num_bags)
    for num in range(1,num_bags+1):
        # Run bagged trees
        out_of_bag_error_35[num-1], test_error_35[num-1] = bagged_trees(X_train, y_train, X_test, y_test, num)
        # Run single tree
        sdt_train_error_35, sdt_test_error_35 = single_decision_tree(X_train, y_train, X_test, y_test)
    
    plt.figure(2)
    plt.plot(np.arange(1,num_bags+1), out_of_bag_error_35)
    plt.legend(['3 versus 5'])
    plt.xlabel('number of bags')
    plt.ylabel('OOB error')
    plt.show()
    
    print('3 versus 5')
    print('Bagging:')
    print('OOB error:', out_of_bag_error_35[num_bags-1])
    print('test error:', test_error_35[num_bags-1])
    print('Single:')
    print('test error:', sdt_test_error_35)


if __name__ == "__main__":
    main_hw4()


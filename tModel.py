#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:51:21 2019
@author: Enyang
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import tFunctions as tFunc
import matplotlib.pyplot as plt
import numpy as np

def model(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, units_per_layer, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Tensorflow will find the dependent operations and run them first if need.	
    Your cost function uses Z3 as a parameter and you calculated Z3 with the forward_propagation function,
    so Tensorflow will run all these functions for you in the correct order.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = tFunc.create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = tFunc.initialize_parameters(units_per_layer)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    z3 = tFunc.forward_propagation(X, parameters, units_per_layer)
    
    # Cost function: Add cost function to tensorflow graph
    cost = tFunc.compute_cost(z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = tFunc.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(minibatch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        # tf.argmax(z3) returns an array with the indexes of the biggest value within z3 tensor
        # Y is one-hot encoded, so it has one 1 and all other are zero. 
        # pred represents probabilities of classes. 
        # So argmax finds the positions of best prediction and correct value. 
        # After that you check whether they are the same.
        # tf.equal returns a 1D array with 0's and 1's
        correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        # By computing the mean of elements across dimensions of a tensor.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Validation Accuracy:", accuracy.eval({X: X_vali, Y: Y_vali}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


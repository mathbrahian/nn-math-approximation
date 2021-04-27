# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# The training set
x = np.asarray([2.5, 5, 3, 6, 1, 1.5, 4, 3.5, 4.5, 2, 4, 2.5])
y = np.asarray([18, 32.5, 22, 36, 10, 13, 30, 22.5, 32, 18, 26, 20])

# Definition of initial parameters
learning_rate = 0.001
training_epochs = 5000

# Variables
W = tf.Variable(np.random.randn(), name="weight") 
b = tf.Variable(np.random.randn(), name="bias") 

# Definition of settings
linear_regression = lambda x : W*x + b #def linear_regression(x): return W*x + b
#def residual_sum_squares(linear_regression, model): return tf.reduce_mean(tf.square(model - linear_regression))
residual_sum_squares = lambda linear_regression, model : tf.reduce_mean(tf.square(model - linear_regression))

# Train
for epoch in range(training_epochs):
    with tf.GradientTape() as gt:
        min_residual_sum_squares = residual_sum_squares(linear_regression(x), y)
    dW, db = gt.gradient(min_residual_sum_squares, [W,b])
    W.assign_sub(learning_rate*dW)
    b.assign_sub(learning_rate*db)

# Graphic display
plt.plot(x, y, 'ro', label='training set') 
plt.plot(x, linear_regression(x),  label='Linear Regression')
plt.legend() 
plt.savefig("linear_regression.png") # Save image
plt.show()
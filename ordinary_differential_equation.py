# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# The training set
x_0 = np.asarray([0], dtype="float64")[:, np.newaxis] #x_0 = np.zeros((50,1)).astype('float32')
f_0 = np.asarray([2], dtype="float64")[:, np.newaxis] #f_0 = 2*np.ones((50,1)).astype('float32')

x = np.linspace(0, 2, 50, dtype="float64")[:, np.newaxis]
y = x**2+2*x+2

# Definition of initial parameters
learning_rate = 0.01
training_epochs = 10000
h = 1e-5
w = 10

# Variables
W = tf.Variable(tf.random.normal([1,w], dtype=tf.float64))
B = tf.Variable(tf.random.normal([w], dtype=tf.float64))
V = tf.Variable(tf.random.normal([w,1], dtype=tf.float64))

# Neural net
def neural_net(x): 
    return tf.matmul(tf.sigmoid(tf.add(tf.matmul(x, W),B)), V)

# EDO
def derivate(x):
    return (h**(-1))*(neural_net(x+h)-neural_net(x))
def function(x):
    return -x*x+neural_net(x)

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate)
mse = tf.keras.losses.MeanSquaredError() 
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        loss = mse(neural_net(x_0), f_0)  + mse(derivate(x), function(x))
    gradients = tape.gradient(loss, [W, B, V])
    optimizer.apply_gradients(zip(gradients, [W, B, V]))

# Graphic display
plt.plot(x, y, 'ro', label='training set') 
plt.plot(x, neural_net(x),  label='Approximate solution of ODE')
plt.legend() 
plt.savefig("ordinary_differential_equation.png") # Save image
plt.show()
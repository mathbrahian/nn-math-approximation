# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# The training set
x = np.linspace(0, 20, 50, dtype="float64")
y = x*np.sin(x)
x = x[:, np.newaxis]
y = y[:, np.newaxis]

# Definition of initial parameters
learning_rate = 0.1
training_epochs = 10000

# Variables
w = 10
W = tf.Variable(tf.random.normal([1,w], dtype=tf.float64))
B = tf.Variable(tf.random.normal([w], dtype=tf.float64))
V = tf.Variable(tf.random.normal([w,1], dtype=tf.float64))

# Neural net
@tf.function
def neural_net(x):  
    return tf.matmul(tf.sigmoid(tf.add(tf.matmul(x, W), B)), V)

# Train 
optimizer = tf.keras.optimizers.Adam(learning_rate)
mse = tf.keras.losses.MeanSquaredError()
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        loss = mse(neural_net(x), y)
    gradients = tape.gradient(loss, [W, B, V])
    optimizer.apply_gradients(zip(gradients, [W, B, V]))

# Graphic display
plt.plot(x, y, 'ro', label='training set') 
plt.plot(x, neural_net(x),  label='function approximation R->R')
plt.legend() 
plt.savefig("function_approximation_R_to_R.png") # Save image
plt.show()
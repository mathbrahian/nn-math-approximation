# Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


with tf.device('/gpu:0'):
    # The trainig set
    n=100
    x=np.linspace(-1, 1, n, dtype="float64") # (n,)
    y=np.linspace(-1, 1, n, dtype="float64") # (n,)
    x_, y_ = np.meshgrid(x, y) # (n,)X(n,) -> (n,n),(n,n)

    _x=np.concatenate((x_), axis=None) # (n^2,)
    _y=np.concatenate((y_), axis=None) # (n^2,)
    _z=_x**2+_y**2

    _x_=_x[:, np.newaxis] # (n,1)
    _y_=_y[:, np.newaxis] # (n,1)
    _z_=_z[:, np.newaxis] # (n,1)

    # Definition of initial parameters
    learning_rate = 0.01
    training_epochs = 50000

    # Variables
    w = 20
    Wx = tf.Variable(tf.random.normal([1,w], dtype=tf.float64))
    Wy = tf.Variable(tf.random.normal([1,w], dtype=tf.float64))
    B = tf.Variable(tf.random.normal([w], dtype=tf.float64))
    V = tf.Variable(tf.random.normal([w,1], dtype=tf.float64))

# Neural net
@tf.function
def neural_net(x,y):  
    return tf.matmul(tf.sigmoid(tf.matmul(x, Wx) + tf.matmul(y, Wy) + B), V)

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate)
mse = tf.keras.losses.MeanSquaredError()
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        loss = mse(neural_net(_x_, _y_), _z_)
    gradients = tape.gradient(loss, [Wx, Wy, B, V])
    optimizer.apply_gradients(zip(gradients, [Wx, Wy, B, V]))

# Graphic display
f = x_**2+y_**2

z_ = np.concatenate(neural_net(_x_,_y_), axis=0).reshape((n,n))
plt.contourf(x_, y_, np.abs(z_-f), cmap='jet')
plt.colorbar()
plt.savefig("function_approximation_R2_to_R.png")
plt.show()


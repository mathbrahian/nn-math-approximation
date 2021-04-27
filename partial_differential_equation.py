import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# The training set
n = 35
m = 50

x = np.linspace(0,1,n,dtype="float64")
y = np.linspace(0,1,n,dtype="float64")

x_, y_ = np.meshgrid(x, y)

_x_ = x_.ravel().reshape((-1, 1))
_y_ = y_.ravel().reshape((-1, 1))

x_1 = np.zeros((1, m),dtype="float64")
y_1 = np.linspace(0, 1, m,dtype="float64")
z_1 = y_1**3

x_2 = np.ones((1, m),dtype="float64")
y_2 = np.linspace(0, 1, m,dtype="float64")
z_2 = (1+y_2**3)*np.exp(-1)

x_3 = np.linspace(0, 1, m,dtype="float64")
y_3 = np.zeros((1, m),dtype="float64")
z_3 = x_3*np.exp(-x_3)

x_4 = np.linspace(0, 1, m, dtype="float64")
y_4 = np.ones((1, m), dtype="float64")
z_4 = np.exp(-x_4)*(x_4+1)

_x0=np.concatenate((x_1,x_2,x_3,x_4), axis=None)
_y0=np.concatenate((y_1,y_2,y_3,y_4), axis=None)
_z0=np.concatenate((z_1,z_2,z_3,z_4), axis=None)

_x0_=_x0[:, np.newaxis]
_y0_=_y0[:, np.newaxis]
_z0_=_z0[:, np.newaxis]

# Definition of initial parameters
learning_rate = 0.01
training_epochs = 50000
h = 1e-5
w = 50

# Variables
Wx = tf.Variable(tf.random.normal([1,w],dtype=tf.float64))
Wy = tf.Variable(tf.random.normal([1,w],dtype=tf.float64))
B = tf.Variable(tf.random.normal([w],dtype=tf.float64))
V = tf.Variable(tf.random.normal([w,1],dtype=tf.float64))

# Neural net
def neural_net(x,y): 
    return tf.matmul(tf.sigmoid(tf.add(tf.add(tf.matmul(x, Wx), tf.matmul(y, Wy)),B)), V)

#PDE
def function(x,y):
    return tf.exp(-x)*(x-2+y**3+6*y)

def laplace(x,y):
    return (h**(-2))*(neural_net(x+h,y)+neural_net(x,y+h)-4*neural_net(x,y)+neural_net(x-h,y)+neural_net(x,y-h))

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate)
mse = tf.keras.losses.MeanSquaredError() 
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        loss = mse(neural_net(_x0_, _y0_), _z0_)  + mse(laplace(_x_,_y_), function(_x_,_y_))
    gradients = tape.gradient(loss, [Wx, Wy, B, V])
    optimizer.apply_gradients(zip(gradients, [Wx, Wy, B, V]))

# Graphic display
def f(x, y):
    return np.exp(-x)*(x+y**3)

z_=np.concatenate(neural_net(_x_,_y_), axis=0).reshape((n,n))
plt.contourf(x_, y_, np.abs(z_-f(x_,y_)), cmap='jet')
plt.colorbar()
plt.savefig("partial_differential_equation.png")
plt.show()
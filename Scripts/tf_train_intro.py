import numpy as np
import tensorflow as tf

coefficients = np.array([[1.], [-20.], [25.]])

w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32, shape=[3, 1])
cost = tf.add(tf.add(tf.multiply(x[0][0], w**2), tf.multiply(x[1][0], w)), x[2][0])

# the last computation of cost could also be performed with regular math syntax as long as w has been initialized as a variable
# cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # 0.01 is the learning rate

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train, feed_dict={x: coefficients})
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict={x: coefficients})

print(session.run(w))

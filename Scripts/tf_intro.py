# import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)  # tf way to write w^20-10w + 25

# the last computation of cost could also be performed with regular math syntax as long as w has been initialized as a variable
# cost = w**2-10*w+25

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # 0.01 is the learning rate

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train)
print(session.run(w))

for i in range(1000):
    session.run(train)

print(session.run(w))

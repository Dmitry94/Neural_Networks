"""
    Simples things from TensorFlow off tutorial.
"""

import tensorflow as tf

# No values, because value on the OUT
c_node1 = tf.constant(3.0)
c_node2 = tf.constant(4.0)
print '1: ', (c_node1, c_node2)

# Value here, because now we are running graph
session = tf.Session()
print '2: ', session.run(c_node1)

# Now we promise to give values latter
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
s = a + b * 3
print '3: ', session.run(s, {a: 3.0, b: 1.0})

# We need to modify some params in the graph
# For this we have variable
W = tf.Variable([3.0])
b = tf.Variable([0.5])
x = tf.placeholder(tf.float32)
lin_model = W * x + b

# We need to init all global variables
init = tf.global_variables_initializer()
session.run(init)

# Now we can evaluate lin model
print '4: ', session.run(lin_model, {x: [1, 2.5, 3.6]})

# But we need loss and labels
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(lin_model - y)
loss = tf.reduce_sum(squared_deltas)
print '5: ', session.run(loss, {x: [1, 2, 3], y:[0, -1, -2]})

# We can make loss equal to zero, if
# assign to the variables good values
fix_w = tf.assign(W, [-1.0])
fix_b = tf.assign(b, [1.0])
session.run([fix_w, fix_b])
print '6: ', session.run(loss, {x: [1, 2, 3], y:[0, -1, -2]})

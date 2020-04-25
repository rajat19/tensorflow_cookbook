# Tensors
#----------------------------------
#
# This function introduces various ways to create
# tensors in TensorFlow

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Introduce tensors in tf

my_tensor = tf.zeros([1,20])
print(my_tensor)

# Declare a variable
my_var = tf.Variable(tf.zeros([1,20]))
print(my_var)

# Different kinds of variables
row_dim = 2
col_dim = 3

# Zero initialized variable
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
print(zero_var)

# One initialized variable
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))
print(ones_var)

# shaped like other variable
zero_similar = tf.Variable(tf.zeros_like(zero_var))
print(zero_similar)

ones_similar = tf.Variable(tf.ones_like(ones_var))
print(ones_similar)

# Fill shape with a constant
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))

# Create a variable from a constant
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# This can also be used to fill an array:
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Sequence generation
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end

sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end

# Random Numbers

# Random
rnorm_var = tf.random.normal([row_dim, col_dim], mean=0.0, stddev=1.0)
runif_var = tf.random.uniform([row_dim, col_dim], minval=0, maxval=4)

print(rnorm_var)
print(runif_var)

# Create variable
my_var = tf.Variable(tf.zeros([1,20]))

# Initialize graph writer:
writer = tf.summary.create_file_writer("/tmp/variable_logs")

with writer.as_default():
    for step in range(100):
        # other model code would go here
        tf.summary.scalar("my_metric", 0.5, step=step)
        writer.flush()

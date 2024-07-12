import tensorflow as tf

# Initialize profiler
tf.profiler.experimental.start(logdir='path/to/logs')

# Define and run your TensorFlow operations
a = tf.constant(1)
b = tf.constant(2)
c = a + b
print(c)

# Stop profiler
tf.profiler.experimental.stop()

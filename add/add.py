import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
addition = a + b
add_n_double = 2 * addition

sess = tf.Session()

print(sess.run(add_n_double, {a:[1.0, 3.0], b:[3.0, 2.0]}))


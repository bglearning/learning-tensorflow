import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
pa = tf.Print(a, [a], message="a: ")
pb = tf.Print(b, [b], message="b: ")
addition = pa + pb
p_add = tf.Print(addition, [addition], message="Sum: ")
add_n_double = 2 * p_add
p = tf.Print(add_n_double, [add_n_double], message="Final: ")

sess = tf.Session()

sess.run(p, {a:[1.0, 3.0], b:[3.0, 2.0]})

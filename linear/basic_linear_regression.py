import tensorflow as tf

logs_path = 'output/basic_linear_regression'

W = tf.Variable([1.0], tf.float32, name='Weights')
b = tf.Variable([1.0], tf.float32, name='Bias')

x = tf.placeholder(tf.float32, name='X')
y = tf.placeholder(tf.float32, name='Y')

with tf.name_scope('LinearModel'):
    linear_model = W*x + b

input_xs = [1.0, 2.0, 3.0]
output_ys = [3.0, 5.0, 7.0]

with tf.name_scope('Loss'):
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

with tf.name_scope('GD'):
    optimizer = tf.train.GradientDescentOptimizer(0.01) 
    train = optimizer.minimize(loss)

tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    for i in range(1000):
        _, summary = sess.run([train, merged_summary_op], {x: input_xs, y: output_ys})
        summary_writer.add_summary(summary, i)

    final_W, final_b, final_loss = sess.run([W, b, loss], {x: input_xs, y: output_ys})
    print('Final W: {}, b: {}, loss: {}'.format(final_W, final_b, final_loss))


import numpy as np
import tensorflow as tf

## Data
XOR_X = [[0.,0.], [0.,1.], [1.,0.], [1.,1.]]
XOR_Y = [[0.], [1.], [1.], [0.]]

## Network Params
num_of_hidden_nodes = 4
num_of_inputs = 2
num_of_steps = 50000
display_step = 1000
learning_rate = 0.2

## tf Graph inputs
X = tf.placeholder(tf.float32, [None,2])
Y = tf.placeholder(tf.float32, [None,1])

## Weights and biases
weights = {
        'W1':tf.Variable(tf.random_normal([num_of_inputs, num_of_hidden_nodes]), name="W1"),
        'W2':tf.Variable(tf.random_normal([num_of_hidden_nodes, 1]), name="W2")
}

biases = {
        'b1': tf.Variable(tf.random_normal([num_of_hidden_nodes])),
        'out': tf.Variable(tf.random_normal([1]))
}

def neural_net(x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['W1']), biases['b1']))
    out_layer = tf.add(tf.matmul(layer_1, weights['W2']), biases['out'])
    return out_layer

logits = neural_net(X)
prediction = tf.sigmoid(logits)

## Loss and optimizer

# For some reason
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# doesn't work. The loss comes out as 0.0 (?)

loss_op = - tf.reduce_mean( (Y * tf.log(prediction)) + (1 - Y) * tf.log(1.0 - prediction)  )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_of_steps + 1):
        sess.run(train_op, feed_dict={X: XOR_X, Y: XOR_Y})

        if step % display_step == 0:
            loss = sess.run(loss_op, feed_dict={X: XOR_X, Y: XOR_Y})
            pred = np.array(sess.run(prediction, feed_dict={X: XOR_X, Y: XOR_Y})).ravel()
            print("Step {}, Loss: {}, pred: {}".format(step, loss, pred))

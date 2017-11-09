import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf

i_sess = tf.InteractiveSession()
DATA_FILE = 'winequality-white.csv'
MODEL_DIR = 'output/manual/'
LEARNING_RATE = 0.0
NUM_OF_FEATURES = 3

def model_fn(features, labels, mode):
    x = features['x']

    init_vals = tf.constant(np.random.rand(NUM_OF_FEATURES + 1, 1))
    #W = tf.get_variable("W", initializer=init_vals, dtype=tf.float64)
    W = tf.get_variable("W", [NUM_OF_FEATURES + 1, 1], dtype=tf.float64)

    y = tf.matmul(x, W)
    y = tf.reshape(y, [-1])

    #loss = tf.losses.mean_squared_error(labels, y)

    #check_op = tf.add_check_numerics_ops()

    #loss = tf.reduce_mean(tf.square(labels - y))
    #loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, y)))
    loss = tf.reduce_sum(tf.squared_difference(labels, y))

    #sess = tf.Session()
    #sess.run(check_op)

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y,
            loss=loss,
            train_op=train)

def main():
    df = pd.read_csv(DATA_FILE, sep=';')

    X = df.drop('quality', axis=1).values
    global NUM_OF_FEATURES
    NUM_OF_FEATURES = X.shape[1]
    y = df['quality'].values.astype(np.float64, copy=False)

    #min_max_scaler = preprocessing.MinMaxScaler()
    #X = min_max_scaler.fit_transform(X)
    X = np.c_[X, np.ones(X.shape[0])]

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_of_instances = x_train.shape[0]

    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=num_of_instances, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=num_of_instances, num_epochs=1, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_test}, y_test, batch_size=x_test.shape[0], num_epochs=1, shuffle=False)

    estimator.train(input_fn=input_fn, steps=100)

    print("Training Complete!")

    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r"% train_metrics)
    print("eval metrics: %r"% eval_metrics)

if __name__ == '__main__':
    main()


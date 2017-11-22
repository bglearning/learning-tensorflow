import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

DATA_FILE = 'winequality-white.csv'

def main():
    df = pd.read_csv(DATA_FILE, sep=';')

    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    feature_columns = [tf.feature_column.numeric_column("x", shape=[X.shape[1]])]

    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_of_instances = x_train.shape[0]

    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=num_of_instances, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=1, num_epochs=1, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_test}, y_test, batch_size=1, num_epochs=1, shuffle=False)

    estimator.train(input_fn=input_fn, steps=1000)

    print("Training Complete!")

    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r"% train_metrics)
    print("eval metrics: %r"% eval_metrics)

if __name__ == '__main__':
    main()


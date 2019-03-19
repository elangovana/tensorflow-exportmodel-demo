import os

import sys
import tensorflow as tf
import logging


def export_model_ckpt(sess, outputdir=None):
    outputdir = outputdir or os.path.join(os.path.dirname(__file__), ".")

    saver = tf.train.Saver()
    save_path = saver.save(sess, outputdir)

    logging.info("Model saved to in ckpt format {}".format(save_path))


def run_linear_regression(gpus: list, outputdir=None):
    outputdir = outputdir or os.path.join(os.path.dirname(__file__), ".")
    devices =  ['/device:GPU:{}'.format(g) for g in gpus]
    strategy = tf.contrib.distribute.MirroredStrategy(devices)
    config = tf.estimator.RunConfig(
        train_distribute=strategy, eval_distribute=strategy)
    regressor = tf.estimator.LinearRegressor(
        feature_columns=[tf.feature_column.numeric_column('feats')],
        optimizer='SGD',
        config=config)
    regressor.train(input_fn=input_fn, steps=100)
    regressor.export_savedmodel(outputdir, input_fn)

def input_fn():
  return tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.])).repeat(10000).batch(10)

def run(gpus: list):
    """

    :param gpus: a list of Gpu device numbers , e.g [0,1]
    """
    # Creates a graph.
    c = []
    for d in ['/device:GPU:{}'.format(g) for g in gpus]:
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
        sum = tf.add_n(c)

    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(sum))
    export_model_ckpt(sess)



if __name__ == '__main__':
    run_linear_regression([0,1])
    # args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s %(name)s %(levelname)s %(process)d/%(threadName)s - %(message)s')

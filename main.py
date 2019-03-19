import os

import sys
import tensorflow as tf
import logging

from tensorflow import Graph, Session
from tensorflow.core.protobuf import meta_graph_pb2
# from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary


def export_model_ckpt(sess, outputdir=None):
    outputdir = outputdir or os.path.join(os.path.dirname(__file__), ".")

    saver = tf.train.Saver()
    save_path = saver.save(sess, outputdir)

    logging.info("Model saved to in ckpt format {}".format(save_path))


def export_model_for_serving(outputdir, estimator):
    export_path_base = os.path.join(outputdir, "linear_model")
    model_version = "1.0"
    export_path = os.path.join(export_path_base, model_version)
    logging.info('Exporting trained model to {}'.format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # TODO: build correct spec
    tensor_info_input = meta_graph_pb2.TensorInfo()
    tensor_info_input.name = "Input"
    tensor_info_input.dtype = 1
    tensor_info_output = meta_graph_pb2.TensorInfo()
    tensor_info_output.name = "Output"
    tensor_info_output.dtype = 1

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': tensor_info_input},
            outputs={'y': tensor_info_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    with Graph().as_default():
        with Session().as_default() as sess:
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_linear':
                        prediction_signature,
                })

            # export the model
            # builder.save(as_text=True)
            builder.save(as_text=False)
    print('Done exporting!')
    import_to_tensorboard(export_path, os.path.join(outputdir, "graph"))


def import_to_tensorboard(model_dir, log_dir):
    """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

    Args:
      model_dir: The location of the protobuf (`pb`) model to visualize
      log_dir: The location for the Tensorboard log to begin visualization from.

    Usage:
      Call this function with your model location and desired log directory.
      Launch Tensorboard by pointing it to the log directory.
      View your imported `.pb` model as a graph.
    """

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)

        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print("Model Imported. Visualize by running: "
              "tensorboard --logdir={}".format(log_dir))


def run_linear_regression(gpus: list, outputdir=None):
    outputdir = outputdir or os.path.join(os.path.dirname(__file__), ".")
    checkpoint_dir = os.path.join(outputdir, "checkpoint")

    # TF Eager Execution
    tf.enable_eager_execution()

    # Runs the op.
    devices = ['/device:GPU:{}'.format(g) for g in gpus]

    strategy = tf.contrib.distribute.MirroredStrategy(devices)
    config = tf.estimator.RunConfig(
        train_distribute=strategy, eval_distribute=strategy)

    regressor = tf.estimator.LinearRegressor(
        feature_columns=[tf.feature_column.numeric_column('features')],
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=.0001),
        model_dir=checkpoint_dir,
        config=config)
    regressor.train(input_fn=input_fn, steps=5)

    results = regressor.predict(input_fn)
    print(results)

    regressor.export_savedmodel(checkpoint_dir, serving_input_fn)
    # Creates a session with log_device_placement set to True.

    import_to_tensorboard(checkpoint_dir, os.path.join(outputdir, "graph"))


def serving_input_fn():
    feature_spec = {"input": tf.FixedLenFeature(dtype=tf.float32, shape=[1])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def input_fn():
    return tf.data.Dataset.from_tensors(({"features": [1.]}, [1.])).repeat(10000).batch(100)


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
    run_linear_regression([0, 1])
    # args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s %(name)s %(levelname)s %(process)d/%(threadName)s - %(message)s')

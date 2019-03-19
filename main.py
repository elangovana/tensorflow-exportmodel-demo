import os

import sys
import tensorflow as tf
import logging

from tensorflow import Graph, Session
from tensorflow.core.protobuf import meta_graph_pb2


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
            builder.save()
    print('Done exporting!')
    write_model_graph(export_path, os.path.join(outputdir, "graph"))

def write_model_graph(model_dir, log_dir):
    import tensorflow as tf
    import sys
    from tensorflow.python.platform import gfile

    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.python.util import compat

    with tf.Session() as sess:
        model_filename = os.path.join(model_dir, 'saved_model.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            # print(sm)
            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            # graph_def = tf.GraphDef()
            # graph_def.ParseFromString(sm.meta_graphs[0])
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    LOGDIR = log_dir
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)


def run_linear_regression(gpus: list, outputdir=None):
    outputdir = outputdir or os.path.join(os.path.dirname(__file__), ".")
    checkpoint_dir = os.path.join(outputdir, "checkpoint")
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

    export_model_for_serving(outputdir, regressor)


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

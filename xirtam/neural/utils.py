"""
TODO(mitch): write doc
"""
import keras.backend as K
import tensorflow as tf


def config_tf():
    """
    TODO(mitch): write doc
    """
    # reduce TF verbosity
    tf.logging.set_verbosity(tf.logging.FATAL)

    # prevent from allocating all memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    K.set_session(sess)

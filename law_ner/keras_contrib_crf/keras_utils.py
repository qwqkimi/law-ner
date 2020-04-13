import keras
import tensorflow as tf

if keras.__name__ == 'keras':
    is_tf_keras = False
elif keras.__name__ == 'tensorflow.keras':
    is_tf_keras = True
else:
    raise KeyError('Cannot detect if using keras or tf.keras.')


def to_tuple(shape):
    """This functions is here to fix an inconsistency between keras and tf.keras.
    In tf.keras, the input_shape argument is an tuple with `Dimensions` objects.
    In keras, the input_shape is a simple tuple of ints or `None`.
    We'll work with tuples of ints or `None` to be consistent
    with keras-team/keras. So we must apply this function to
    all input_shapes of the build methods in custom layers.
    """
    if is_tf_keras:
        import tensorflow as tf
        return tuple(tf.TensorShape(shape).as_list())
    else:
        return shape


def tf_version():
    return tf.version.VERSION.split('.')[0]

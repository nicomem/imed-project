import tensorflow as tf
import tensorflow.keras as k

from sklearn.preprocessing import MinMaxScaler

class MobileNetPreprocess(k.layers.Layer):
    def call(self, inputs):
        inputs = MinMaxScaler((0, 255))

        # call mobilenet preprocess
        inputs = keras.applications.mobilenet_v2.preprocess_input(inputs)

        return inputs

def model_transfer_mobilenet():
    '''
    Build a model with transfer learning using MobileNetV2.

    The model will take as input the (3DT1, T1, FLAIR) images
    corresponding to a slice and return a wmh image.
    '''

    # 3 images of any shape
    input_shape = (None, None, 3)

    inputs = k.Input(shape=input_shape)
    minmaxscaler = k.layers.Lambda(MinMaxScaler((0, 255)), name='MinMaxScaler(0,255)')(inputs)
    preprocess = k.applications.mobilenet_v2.preprocess_input(minmaxscaler)

    return k.Model(inputs, preprocess)

    # preprocess = MobileNetPreprocess()(inputs)

    # Use transfer learning and disable training on these weights
    mobilenet = keras.applications.MobileNetV2(include_top=False)
    for layer in mobilenet.layers:
        layer.trainable = False
    mobilenet = mobilenet(preprocess)

    pool = layers.GlobalMaxPool2D()(mobilenet)
    dense1 = layers.Dense(32, activation='relu')(pool)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    dropout = layers.Dropout(0.25)(dense2)
    outputs = layers.Dense(1, activation='sigmoid')(dropout)

    model2 = keras.Model(inputs, outputs)
    model2.summary()

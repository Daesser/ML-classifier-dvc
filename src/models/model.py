import tensorflow as tf
import tensorflow_hub as hub
import yaml

model_spec = yaml.safe_load(open('params.yaml'))['model']
params = yaml.safe_load(open('params.yaml'))['prepare']
image_width = int(params['width'])
image_width = int(params['height'])


def hub_model_res(num_classes):
    front = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    back = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    side = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    keras_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", trainable=False)
    out1 = keras_layer(front)
    out2 = keras_layer(back)
    out3 = keras_layer(side)
    merged = tf.keras.layers.Concatenate()([out1, out2, out3])
    merged = tf.keras.layers.Flatten()(merged)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(merged)
    model = tf.keras.models.Model(inputs=[front, back, side], outputs=output)
    model.compile(optimizer=model_spec['optimizer'], loss=model_spec['loss'], metrics=model_spec["metrics"])
    return model


def hub_model_ef(num_classes):
    front = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    back = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    side = tf.keras.layers.Input(shape=(image_width, image_width, 3))
    keras_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2", trainable=False)
    out1 = keras_layer(front)
    out2 = keras_layer(back)
    out3 = keras_layer(side)
    merged = tf.keras.layers.Concatenate()([out1, out2, out3])
    merged = tf.keras.layers.Flatten()(merged)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(merged)
    model = tf.keras.models.Model(inputs=[front, back, side], outputs=output)
    model.compile(optimizer=model_spec['optimizer'], loss=model_spec['loss'], metrics=model_spec["metrics"])
    return model


def get_model(model, num_classes):
    if model == 'ef':
        return hub_model_ef(num_classes)
    if model == 'resnet':
        return hub_model_res(num_classes)

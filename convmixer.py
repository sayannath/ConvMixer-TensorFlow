"""
# Image Classification using ConvMixer

> Dataset: CIFAR-100

Paper Link: https://openreview.net/pdf?id=TVHS5Y4dNvM
"""

# !pip install -q tensorflow-addons

"""## Imports"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

SEEDS = 42

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

"""## Load the `CIFAR-100` dataset
In this example, we will use the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset.
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""## Define hyperparameters"""

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
IMG_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 100
EPOCHS = 25
LOG_DIR = "./train-logs.csv"

"""## Define the image preprocessing function"""


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label


"""## Data augmentation"""

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical"
        ),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

"""## Convert the data into TensorFlow `Dataset` objects"""

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

"""## Define the data pipeline"""

# Training Pipeline
pipeline_train = (
    train_ds.shuffle(BATCH_SIZE * 100)
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

# Validation Pipeline
pipeline_validation = (
    validation_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

"""## Visualise the training samples"""

image_batch, label_batch = next(iter(pipeline_train))

plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.axis("off")

"""## Build the ConMixer model

![](https://i.imgur.com/Yd7gpMP.png) 
"""


def build_convmixer_model(num_filters=256, depth=8, kernel_size=2, patch_size=2):

    # Input layer
    input_layer = keras.Input((IMG_SIZE, IMG_SIZE, 3))

    # Patch embedding layer
    patch_layer = keras.layers.Conv2D(
        num_filters, kernel_size=patch_size, strides=patch_size
    )(input_layer)
    act_1 = tfa.layers.GELU()(patch_layer)
    bn_1 = keras.layers.BatchNormalization()(act_1)

    bn = bn_1

    # ConvMixer layer
    for i in range(depth):
        # Residual connection
        depth_conv_1 = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size, padding="same"
        )(bn_1)
        act_2 = tfa.layers.GELU()(depth_conv_1)
        bn_2 = keras.layers.BatchNormalization()(act_2)

        # Add skip connection
        add_1 = keras.layers.add([bn_2, bn])

        # Pointwise convolution layer
        conv_1 = keras.layers.Conv2D(num_filters, kernel_size=1, padding="same")(add_1)
        act_3 = tfa.layers.GELU()(conv_1)
        bn_3 = keras.layers.BatchNormalization()(act_3)

        bn = bn_3

    avg_pool = keras.layers.GlobalAvgPool2D()(bn_3)

    # Output layer
    output_layer = keras.layers.Dense(NUM_CLASSES, activation="softmax")(avg_pool)

    model = keras.Model(input_layer, output_layer, name="ConvMixer-Model")
    return model


model = build_convmixer_model(num_filters=256, depth=8, kernel_size=2, patch_size=2)
model.summary()

"""## Define optimizer and loss"""

optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = keras.losses.CategoricalCrossentropy()

"""## Compile the model"""

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

"""## Set up callbacks"""

train_callbacks = [
    keras.callbacks.CSVLogger(LOG_DIR),
    keras.callbacks.TensorBoard(histogram_freq=1),
]

history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=pipeline_validation,
    callbacks=train_callbacks,
)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()

"""## Evaluate the model"""

accuracy = model.evaluate(pipeline_validation)[1] * 100
print("Accuracy: {:.2f}%".format(accuracy))

import tensorflow as tf
import tensorflow_datasets as tfds
import math


BATCHSIZE = 32


# Normalize from 0-255 to 0-1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255

    return images, labels


dataset, metadata = tfds.load("mnist", as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]

class_names = [
    "Cero",
    "Uno",
    "Dos",
    "Tres",
    "Cuatro",
    "Cinco",
    "Seis",
    "Siete",
    "Ocho",
    "Nueve",
]

num_train_examples = metadata.splits["train"].num_examples
num_test_examples = metadata.splits["test"].num_examples


train_dataset = (
    train_dataset.map(normalize).repeat().shuffle(num_train_examples).batch(BATCHSIZE)
)
test_dataset = test_dataset.map(normalize).batch(BATCHSIZE)

# Initialize ANN and set layers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compile and fit
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE),
)

# Predict values using testing dataset
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples / 32)
)

print("Resultado en las pruebas: ", test_accuracy)

model.save("model.h5")

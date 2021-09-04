from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']

iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=3)

train_dataset = tf.data.Dataset.from_tensor_slices((iris_data_train, iris_target_train))
test_dataset = tf.data.Dataset.from_tensor_slices((iris_data_test, iris_target_test))


BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 20

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 4)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_dataset, epochs=50)

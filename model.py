# linear regression model built with tensorflow

from gc import callbacks
import os
import pickle
from tabnanny import verbose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_df = pd.read_csv("./sample.csv")
train_df.head()

data = train_df.dropna()
# print(data)

train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

sns.pairplot(train_data[['distance', 'time']], diag_kind='kde')

# train_data.describe().transpose()

train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop('time')
test_labels = test_features.pop('time')

# train_data.describe().transpose()[['mean', 'std']]

# normalize the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

# regression model


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    return model


dist = np.array(train_features['distance'])
dist_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
dist_normalizer.adapt(dist)
model = build_and_compile_model(dist_normalizer)
# model.summary()
# print(model)

# # saving the model using check points
# checkpoint_path = "./cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # creating a callback that saves the model
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_weights_only=True, verbose=1)

# training the model
history = model.fit(
    train_features['distance'],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100,)
# callbacks=[cp_callback])

# save the model
# model.save('my_model.h5')
model.save('new_model')

# Evaluate the restored model
new_model = tf.keras.models.load_model('new_model')

# loss, acc = new_model.evaluate(test_features, test_labels)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

new_model.summary()

print(new_model.predict(test_labels).shape)
# with open('model_saved', 'wb') as f:
#     pickle.dump(model, f)
# with open('model_saved', 'rb') as f:
#     df = pickle.load(f)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [time]')
    plt.legend()
    plt.grid(True)


plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = model.predict(x)


def plot_dist(x, y):
    plt.scatter(train_features['distance'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('distance')
    plt.ylabel('time')
    plt.legend()


plot_dist(x, y)

test_results = {}

test_results['model'] = model.evaluate(
    test_features['distance'], test_labels,
    verbose=0)

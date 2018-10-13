import joblib
import tensorflow as tf
from tensorflow import keras

pickle_file = './FERDataset/FER.joblib'

with open(pickle_file, 'rb') as f:
  save = joblib.load(f)

  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']

  del save

  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 48
num_labels = 7
num_channels = 1 # grayscale

model = keras.Sequential([
  keras.layers.Reshape((1,48,48), input_shape=((48,48))),
  keras.layers.Conv2D(filters=64,
                      kernel_size=5,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=64,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.MaxPooling2D(pool_size=(3,3),
                            strides=(2,2),
                            padding="same",
                            data_format='channels_first'),
  keras.layers.Dropout(0.25),
  keras.layers.Conv2D(filters=128,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=128,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.MaxPooling2D(pool_size=(3,3),
                            strides=(2,2),
                            padding="same",
                            data_format='channels_first'),
  keras.layers.Dropout(0.25),
  keras.layers.Conv2D(filters=256,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=256,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=256,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.MaxPooling2D(pool_size=(3,3),
                            strides=(2,2),
                            padding="same",
                            data_format='channels_first'),
  keras.layers.Dropout(0.25),
  keras.layers.Conv2D(filters=512,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=512,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.Conv2D(filters=512,
                      kernel_size=3,
                      data_format='channels_first',
                      strides=1,
                      padding='same',
                      use_bias=True,
                      activation=keras.activations.relu,
                      bias_initializer=keras.initializers.TruncatedNormal,
                      kernel_initializer=keras.initializers.TruncatedNormal),
  keras.layers.BatchNormalization(axis=1),
  keras.layers.MaxPooling2D(pool_size=(3,3),
                            strides=(2,2),
                            padding="same",
                            data_format='channels_first'),
  keras.layers.Dropout(0.25),
  keras.layers.Flatten(),
  keras.layers.Dense(units=1024,
                     activation=keras.activations.relu,
                     use_bias=True,
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.04),
                     bias_initializer=keras.initializers.Constant(value=0.1),
                     kernel_regularizer=keras.regularizers.l2(0.004)),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(units=1024,
                     activation=keras.activations.relu,
                     use_bias=True,
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.04),
                     bias_initializer=keras.initializers.Constant(value=0.1),
                     kernel_regularizer=keras.regularizers.l2(0.004)),
  keras.layers.Dropout(0.25),
  keras.layers.Dense(units=7,
                     activation=keras.activations.softmax,
                     use_bias=True,
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=1/192.0),
                     bias_initializer=keras.initializers.Constant(value=0.0))
])

model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=1)

model.compile(keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from keras.utils import plot_model
plot_model(model, to_file='results/vgglike.png')


model.fit(train_dataset,
          train_labels,
          epochs=400,
          batch_size=512,
          validation_data=(valid_dataset, valid_labels),
          verbose=1,
          callbacks=[early_stop])

test_loss, test_acc = model.evaluate(test_dataset, test_labels)
print('Test accuracy: ', test_acc)


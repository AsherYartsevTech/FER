import joblib
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

image_size = 48
num_classes = 7
num_channels = 1

def conv_pool_cnn(inputs):
  x = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(inputs)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(96, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

  x = Conv2D(192, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(192, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(192, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

  x = Conv2D(192, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(192, kernel_size=(1,1), activation='relu')(x)
  x = BatchNormalization(axis=-1)(x)
  x = Conv2D(num_classes, kernel_size=(1,1))(x)

  x = GlobalAveragePooling2D()(x)
  x = Activation(activation='softmax')(x)

  model = Model(inputs, x, name='conv_pool_cnn')
  return model

def compile_and_train(model, num_epochs, train_dataset, train_labels, valid_dataset, valid_labels):
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])

    filepath = './weights/' + model.name + '.{epoch:3d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='loss',
                                 verbose=0,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)
    tensor_board = TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=512)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

    history = model.fit(x=train_dataset,
                        y=train_labels,
                        batch_size=512,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=[checkpoint, tensor_board, early_stopping],
                        validation_data=(valid_dataset,valid_labels))

    return history


if __name__ == "__main__":
  pickle_file = './FERDataset/FER4dims.joblib'

  with open(pickle_file, 'rb') as f:
    save = joblib.load(f)

    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    train_labels = to_categorical(train_labels, num_classes)
    valid_labels = to_categorical(valid_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    del save

  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


  input_shape = train_dataset[0,:,:,:].shape
  model_input = Input(shape=input_shape)

  conv_pool_cnn_model = conv_pool_cnn(model_input)
  _ = compile_and_train(conv_pool_cnn_model, 200, train_dataset, train_labels, valid_dataset, valid_labels)

  test_loss, test_acc = conv_pool_cnn_model.evaluate(test_dataset, test_labels)
  print("Test accuracy: ", test_acc)
  

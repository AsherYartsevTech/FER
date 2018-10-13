import joblib
from keras.models import Model, Input
from keras.layers import Average
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from convPoolCnn import conv_pool_cnn
from allCnn import all_cnn
from ninCnn import nin_cnn
from vgglike4ensemble import vgglike_cnn
from dense4ensemble import dense_cnn

image_size = 48
num_classes = 7
num_channels = 1

def ensemble(models, model_input):
  outputs = [model.outputs[0] for model in models]
  y = Average()(outputs)

  model = Model(model_input, y, name='ensemble')
  return model

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
  all_cnn_model = all_cnn(model_input)
  nin_cnn_model = nin_cnn(model_input)
  vgglike_cnn_model = vgglike_cnn(model_input)
  dense_cnn_model = dense_cnn(model_input)

  conv_pool_cnn_model.load_weights('weights/conv_pool_cnn. 24-0.0665.hdf5')
  all_cnn_model.load_weights('weights/all_cnn. 14-0.1041.hdf5')
  nin_cnn_model.load_weights('weights/nin_cnn. 49-1.0130.hdf5')
  vgglike_cnn_model.load_weights('weights/vgglike_cnn. 29-0.1168.hdf5')
  dense_cnn_model.load_weights('weights/dense_cnn. 54-0.0904.hdf5')

  models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model, vgglike_cnn_model, dense_cnn_model]
  ensemble_model = ensemble(models, model_input)
  ensemble_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc'])

  test_loss, test_acc = ensemble_model.evaluate(test_dataset, test_labels)
  print("Test accuracy: ", test_acc)
  

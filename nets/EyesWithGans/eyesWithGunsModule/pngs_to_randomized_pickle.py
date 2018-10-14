from __future__ import print_function
import imageio
import numpy as np
import os
from six.moves import cPickle as pickle
import time
from functools import reduce
from PIL import Image

data_root = '../FERDataset' 
data_folder = os.path.join(data_root, 'data')

image_size = 48  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

num_classes = 7
np.random.seed(int(time.time() // 1))

data_folders = [os.path.join(data_folder, str(i)) for i in range(num_classes)]

def load_class(folder):
  """Load the data for a single expression label from pngs to normalized 3D array"""
  image_files = os.listdir(folder)
  print('class size: ', len(image_files))
  
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      # normalize image - consider only to divide by pixel depth
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth

      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))

      dataset[num_images, :, :] = image_data
      num_images = num_images + 1

    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def augment_by_mirror(dataset):
  augmented = np.ndarray(shape=(2 * dataset.shape[0], image_size, image_size),
                           dtype=np.float32)
  for i, image in enumerate(dataset):
    augmented[2*i, :, :] = image
    augmented[2*i+1, :, :] = Image.fromarray(image).transpose(Image.FLIP_LEFT_RIGHT)

  return augmented

def maybe_pickle(data_folders, force=False):
  dataset_names = []

  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)

    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_class(folder)
      dataset = augment_by_mirror(dataset)

      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

datasets = maybe_pickle(data_folders)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files):
  train_size,valid_size, test_size = 0, 0, 0
  categories_sizes = []

  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        category_set = pickle.load(f)
        category_size = category_set.shape[0]
        
        categories_sizes.append({
          'train': int(0.7  * category_size),
          'valid': int(0.15 * category_size),
          'test':  category_size - int(0.7  * category_size) - int(0.15 * category_size)
        })

    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  train_size = reduce((lambda r,c: r + c['train']), categories_sizes, 0) 
  valid_size = reduce((lambda r,c: r + c['valid']), categories_sizes, 0) 
  test_size  = reduce((lambda r,c: r + c['test']),  categories_sizes, 0) 

  train_dataset, train_labels = make_arrays(train_size, image_size)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  test_dataset, test_labels = make_arrays(test_size, image_size)

  start_train, start_valid, start_test = 0, 0, 0

  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        category_set = pickle.load(f)
        np.random.shuffle(category_set)

        train_csize = categories_sizes[label]['train']
        valid_csize = categories_sizes[label]['valid']
        test_csize  = categories_sizes[label]['test']

        train_data = category_set[:train_csize, :, :]
        valid_data = category_set[train_csize:train_csize+valid_csize, :, :]
        test_data  = category_set[train_csize+valid_csize:train_csize+valid_csize+test_csize, :, :]
        
        train_dataset[start_train:start_train+train_csize, :, :] = train_data
        valid_dataset[start_valid:start_valid+valid_csize, :, :] = valid_data
        test_dataset[start_test:start_test+test_csize, :, :] = test_data

        train_labels[start_train:start_train+train_csize] = label
        valid_labels[start_valid:start_valid+valid_csize] = label
        test_labels[start_test:start_test+test_csize] = label

        start_train += train_csize
        start_valid += valid_csize
        start_test  += test_csize
                    
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
            
            
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = merge_datasets(datasets)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


pickle_file = os.path.join(data_root, 'FER.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()

except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


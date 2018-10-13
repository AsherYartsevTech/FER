from __future__ import print_function
import imageio
import numpy as np
import os
import joblib
import time
from functools import reduce
from PIL import Image

data_root = './FERDataset' 
train_folder = os.path.join(data_root, 'data', 'train')
valid_folder = os.path.join(data_root, 'data', 'valid')
test_folder = os.path.join(data_root, 'data', 'test')

image_size = 48  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

num_classes = 7
np.random.seed(int(time.time() // 1))

train_folders = [os.path.join(train_folder, str(i)) for i in range(num_classes)]
valid_folders = [os.path.join(valid_folder, str(i)) for i in range(num_classes)]
test_folders = [os.path.join(test_folder, str(i)) for i in range(num_classes)]

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

def augment_by_rotation(dataset):
  augmented = np.ndarray(shape=(3 * dataset.shape[0], image_size, image_size),
                           dtype=np.float32)
  for i, image in enumerate(dataset):
    augmented[3*i, :, :] = image
    augmented[3*i+1, :, :] = Image.fromarray(image).rotate(20,resample=Image.BILINEAR,expand=1).resize((image_size,image_size),resample=Image.BILINEAR)
    augmented[3*i+2, :, :] = Image.fromarray(image).rotate(-20,resample=Image.BILINEAR,expand=1).resize((image_size,image_size),resample=Image.BILINEAR)

  return augmented

def maybe_pickle(data_folders, shouldAugment=False, force=False):
  dataset_names = []

  for folder in data_folders:
    set_filename = folder + '.joblib'
    dataset_names.append(set_filename)

    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      dataset = load_class(folder)

      if shouldAugment:
        print('Mirroring images...')
        dataset = augment_by_mirror(dataset)
        print('rotating images...')
        dataset = augment_by_rotation(dataset)

      print('Pickling %s.' % set_filename)

      try:
        with open(set_filename, 'wb') as f:
          joblib.dump(dataset, f)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, shouldAugment=True)
valid_datasets = maybe_pickle(valid_folders, shouldAugment=False)
test_datasets = maybe_pickle(test_folders, shouldAugment=False)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def split_datasets(pickle_files, shouldSplit=False):
  categories_sizes = []

  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        category_set = joblib.load(f)
        category_size = category_set.shape[0]
      
        if shouldSplit:
          categories_sizes.append({
            'train': int(0.9  * category_size),
            'valid': category_size - int(0.9  * category_size)
          })
        else:
          categories_sizes.append({
            'train': category_size,
            'valid': 0
          })

    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  train_size = reduce((lambda r,c: r + c['train']), categories_sizes, 0)
  valid_size = reduce((lambda r,c: r + c['valid']), categories_sizes, 0)

  train_dataset, train_labels = make_arrays(train_size, image_size)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)

  start_train, start_valid = 0, 0

  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        category_set = joblib.load(f)
        np.random.shuffle(category_set)

        train_csize = categories_sizes[label]['train']
        train_data = category_set[:train_csize, :, :]
        train_dataset[start_train:start_train+train_csize, :, :] = train_data
        train_labels[start_train:start_train+train_csize] = label

        if valid_dataset is not None:
          valid_csize = categories_sizes[label]['valid']
          valid_data = category_set[train_csize:train_csize+valid_csize, :, :]
          valid_dataset[start_valid:start_valid+valid_csize, :, :] = valid_data
          valid_labels[start_valid:start_valid+valid_csize] = label

        start_train += train_csize
        if valid_dataset is not None:
          start_valid += valid_csize
                    
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return train_dataset, train_labels, valid_dataset, valid_labels
            
            
train_dataset, train_labels, _, _ = split_datasets(train_datasets, shouldSplit=False)
valid_dataset, valid_labels, _, _ = split_datasets(valid_datasets, shouldSplit=False)
test_dataset, test_labels, _, _ = split_datasets(test_datasets, shouldSplit=False)

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


pickle_file = os.path.join(data_root, 'FER.joblib')

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
  joblib.dump(save, f)
  f.close()

except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


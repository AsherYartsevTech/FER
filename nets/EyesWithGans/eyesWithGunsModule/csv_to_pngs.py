from __future__ import print_function
import numpy as np
import os
import csv
from PIL import Image
from itertools import islice

data_root = '../FERDataset' 
csv_path = os.path.join(data_root,'fer2013.csv')

image_size = 48  # Pixel width and height.

def str_to_image(image_blob):
  ''' convert a string blob to an image object. '''
  image_string = image_blob.split(' ')
  image_data = np.asarray(image_string, dtype=np.uint8).reshape(image_size,image_size)
  return Image.fromarray(image_data)

num_classes = 7

def maybe_extract(force=False):
  data_folder = os.path.join(data_root, 'data')
  if os.path.isdir(data_folder) and not force:
    print('%s already present - Skipping extraction.' % data_folder)
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % data_folder)
    os.mkdir(data_folder)
    for i in range(num_classes): os.mkdir(os.path.join(data_folder, str(i)))

    with open(csv_path, 'r') as csvfile:
      fer_rows = csv.reader(csvfile, delimiter=',')

      index = 0
      for row in islice(fer_rows, 1, None):
        label = str(row[0])
        file_name = f'fer{index:05}.png'
        image = str_to_image(row[1])
        image_path = os.path.join(data_folder, label, file_name)
        image.save(image_path, compress_level=0)

        index += 1

  data_folders = [
    os.path.join(data_folder, d) for d in sorted(os.listdir(data_folder))
    if os.path.isdir(os.path.join(data_folder, d))]

  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))

  print(data_folders)

maybe_extract()

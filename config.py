import pathlib
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print(full_filename)

car_dir = pathlib.Path('car')

image_count = len(list(car_dir.glob('*/*.jpg')))
print(image_count)

#search("car/")

BMW = list(car_dir.glob('BMW/*'))
test = PIL.Image.open(str(BMW[0]))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  car_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(0):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
import pathlib
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
batch_size = 32
img_height = 180
img_width = 180

car_dir = pathlib.Path('car')

image_count = len(list(car_dir.glob('*/*.jpg')))
print(image_count)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  car_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  car_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
print(len(class_names))

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

  AUTOTUNE = tf.data.experimental.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixels values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image))

  num_classes = 42

  model = Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\n테스트 정확도:', test_acc)


test_num = np.argmax(predictions[1])
print(test_num)
print(class_names[test_num])

import matplotlib.pyplot as plt
import cv2
%matplotlib inline

test_num = plt.imread('./car_test_img/test_img00.jpg')
#test_num = test_num[:,:,0]
#test_num = (test_num > 125) * test_num
#test_num = test_num.astype('float32') / 255.

#test_num = test_num.reshape(1, 180, 180, 1)
plt.imshow(test_num, cmap='Greys', interpolation='nearest');
print('The Answer is ', model.predict_classes(val_ds))


file_path = ('./car_test_img/test_img40.jpg')
from keras.models import load_model
model = load_model('my_model.h5')
image = Image.open(file_path)
image = np.array(image) #convert image to numpy array
print(image.shape)
x_test = np.array([image])
y_predict = model.predict_classes(x_test)
print(y_predict) #[7]
print(class_names[23])
plt.imshow(image)
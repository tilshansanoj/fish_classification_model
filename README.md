# Fish Classification Model
Fish Classification AI model trained by using Tensorflow

First of all, we need to train a model to detect fish from other animals

# Fish Detection Model

In here we are going to do a binary classification by using a pre-trained model called MobileNet


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Step 1: Data Preparation
train_data_dir = 'path\to\train\dataset'
test_data_dir = 'path\to\test\dataset'
image_size = (224, 224)
batch_size = 32

# Apply data augmentation to the training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Normalize the pixel values of the testing dataset
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

# Load and prepare the testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary')

# Load the MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with 256 units and ReLU activation
x = Dense(256, activation='relu')(x)

# Add the final prediction layer with a sigmoid activation for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# Create the model with the MobileNet base and the new prediction layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the MobileNet layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Step 3: Model Training
num_epochs = 7

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)

# Save the trained model
model.save('fish_detection.h5')
```
# Fish Species Classification Model

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# Set the paths to your training and testing datasets
train_dir = ''path\to\train\dataset'
test_dir = 'path\to\test\dataset'

# Extract class names from folder names
class_names = sorted(os.listdir(train_dir))

print("Fish Classes: ",class_names)

# Set the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 8

# Preprocess and augment the training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Preprocess the testing data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names
)

# Load the testing data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names
)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    epochs=10,
    validation_data=test_data,
    validation_steps=test_data.samples // batch_size
)

# Evaluate the model on the testing data
score = model.evaluate(test_data, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model 
model.save('fish_classification.h5')
```
By combining both models we are able predict if the input image has a fish or not, if yes then what is name of the fish species

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
import numpy as np

#Getting the names of the classes from the folder names
train_dir = 'E:\Fish\Species\Training_Set'
class_names = sorted(os.listdir(train_dir))

# Load the fish detection model
fish_detection_model = tf.keras.models.load_model('fish_detection.h5')

# Load the fish classification model
fish_classification_model = tf.keras.models.load_model('fish_classification.h5')

# Load and preprocess the input image
img_path = 'fish.jpg'  # Replace with the actual path of your input image
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Normalize the image

# Perform fish detection prediction
is_not_fish = fish_detection_model.predict(img)

is_not_fish_label = 'Not Fish' if is_not_fish > 0.5 else 'Fish'
print('Image is:', is_not_fish_label)

if is_not_fish < 0.5:
    # Perform fish species classification prediction
    species_prediction = fish_classification_model.predict(img)
    species_index = np.argmax(species_prediction)
    species = class_names[species_index]  # class_names is a list of the fish species names
    print('Fish species:', species)
```

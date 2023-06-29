# Fish Classification Model
Fish Classification AI model trained by using Tensorflow

First of all, we need to train a model to detect fish from other animals

#1. Fish Detection Model

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


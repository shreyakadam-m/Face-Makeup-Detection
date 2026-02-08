# ==========================================
# FACE MAKEUP DETECTION - TRAINING SCRIPT
# ==========================================

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Step 1: Loading Dataset...")

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# Data Augmentation + Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    "dataset/validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("Step 2: Building CNN Model...")

model = Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),

    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Step 3: Training Model...")

history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

print("Step 4: Saving Model...")
model.save("makeup_model.h5")

# ===== ACCURACY GRAPH (TensorFlow safe version) =====
print("Step 5: Creating Accuracy Graph...")

acc = history.history.get('accuracy') or history.history.get('acc')
val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')

plt.plot(acc)
plt.plot(val_acc)
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.savefig("accuracy_graph.png")
plt.show()

print("🎉 PROJECT COMPLETED SUCCESSFULLY 🎉")

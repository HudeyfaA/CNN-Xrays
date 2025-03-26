import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_DIR_NORMAL = r"archive/chest_xray/train/NORMAL"
TRAIN_DIR_PNEUMONIA = r"archive/chest_xray/train/PNEUMONIA"
TEST_DIR_NORMAL = r"archive/chest_xray/test/NORMAL"
TEST_DIR_PNEUMONIA = r"archive/chest_xray/test/PNEUMONIA"

def create_dataset(normal_dir, pneumonia_dir, img_size=256):
    dataset = []
    for img_name in os.listdir(normal_dir):
        img_path = os.path.join(normal_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        dataset.append([img / 255.0, 0])

    for img_name in os.listdir(pneumonia_dir):
        img_path = os.path.join(pneumonia_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        dataset.append([img / 255.0, 1])

    shuffle(dataset)
    return dataset

IMG_SIZE = 256

full_data = create_dataset(TRAIN_DIR_NORMAL, TRAIN_DIR_PNEUMONIA, img_size=IMG_SIZE)
X = np.array([data[0] for data in full_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array([data[1] for data in full_data])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

score = model.evaluate(X_train, y_train)
print(f"Accuracy: {score[1] * 100:.2f}%")

model.save('pneumonia_model.h5')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

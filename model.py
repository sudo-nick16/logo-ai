import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

csv_path = "dataset/dataset.csv"
df = pd.read_csv(csv_path) 
image_paths = df['image_path'].tolist()
labels = df['label'].values
labels_encoded = df['label_encoded'].values
num_classes = 2

img_width = 100
img_height = 100
channels = 1

images = []
for path in image_paths:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    images.append(image)
images = np.array(images)
images = images


# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)

# Step 3: Define Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Step 4: Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

n_samples = X_train.shape[0]
X_train = X_train.reshape(n_samples, img_height, img_width, channels)

datagen.fit(X_train)

# Step 6: Train Model
n_epochs = 10
n_batches = 128

history = model.fit(
        X_train, y_train,
        batch_size=n_batches,
        epochs=n_epochs,
        validation_data=(X_test, y_test),
)

# Step 7: Evaluate Model
evaluation = model.evaluate(X_test, y_test)
print("Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

model.save("model.keras")

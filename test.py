import numpy as np
import pandas as pd
import os
import cv2
from keras.models import load_model

test_images_path = "dataset/train"
dataset = "dataset/dataset.csv"
df = pd.read_csv(dataset)
labels = df['label'].values
encoded = df['label_encoded'].values

mapping = {}
for i in range(len(labels)):
    mapping[encoded[i]] = labels[i]

model = load_model("model.h5")

for img in os.listdir(test_images_path):
    img_path = os.path.join(test_images_path, img)
    curr_img = cv2.imread(img_path, 3)
    curr_img = cv2.resize(curr_img, (100, 100))
    curr_img = np.expand_dims(curr_img, axis=0)
    prediction = model.predict(curr_img)
    print("IMAGE:", img, mapping[np.argmax(prediction)])
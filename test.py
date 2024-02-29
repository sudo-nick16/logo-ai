import numpy as np
import pandas as pd
import os
import cv2
from keras.models import load_model

def assert_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

test_images_path = "dataset/train"
dataset = "dataset/dataset.csv"
df = pd.read_csv(dataset)
labels = df['label'].values
encoded = df['label_encoded'].values

mapping = {}
for i in range(len(labels)):
    mapping[encoded[i]] = labels[i]

model = load_model("model.keras")

correct = 0
incorrect = 0

assert_dir(test_images_path)
for img in os.listdir(test_images_path):
    img_path = os.path.join(test_images_path, img)
    curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    curr_img = cv2.resize(curr_img, (100, 100))
    curr_img = np.expand_dims(curr_img, axis=0)
    prediction = model.predict(curr_img)
    print("IMAGE:", img, "PREDICTED:", mapping[np.argmax(prediction)])
    if mapping[np.argmax(prediction)] in img:
        correct += 1
    else:
        incorrect += 1

print("Correct:", correct)
print("Incorrect:", incorrect)
print("Accuracy:", correct/(correct+incorrect)*100, "%")

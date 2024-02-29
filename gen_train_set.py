import os
import cv2
import pandas as pd

images_path = "images"
train_set_path = "dataset/train"

data = []

for i, dir in enumerate(os.listdir(images_path)):
    label = dir
    label = label.lower().replace(" ", "_")
    dir = os.path.join(images_path, dir)
    file_index = 0

    if os.path.isdir(dir):
        for img_name in os.listdir(dir):
            img_path = os.path.join(dir, img_name)
            ext = img_name.split(".")[-1]
            save_path = os.path.join(train_set_path, f"{label}_{file_index}.{ext}")
            img = cv2.imread(img_path, 3)
            img = cv2.resize(img, (100, 100))
            cv2.imwrite(save_path, img)
            data.append([save_path, label, i])
            file_index += 1

df = pd.DataFrame()
df["image_path"] = [d[0] for d in data]
df["label"] = [d[1] for d in data]
df["label_encoded"] = [d[2] for d in data]
df.to_csv("dataset/dataset.csv", index=False)

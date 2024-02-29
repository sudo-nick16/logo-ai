import cv2
import numpy as np
import os

bg_images_path = "dataset/bg_images"
logos_path= "dataset/logos"
test_dataset_path = "dataset/test"

logos = []

for logo in os.listdir(logos_path):
    logos.append({
        "name": logo.split(".")[0],
        "logo": cv2.imread(os.path.join(logos_path, logo), cv2.IMREAD_UNCHANGED)
    })

for i, image in enumerate(os.listdir(bg_images_path)):
    img = cv2.imread(os.path.join(bg_images_path, image), cv2.IMREAD_UNCHANGED)

    for logo in logos:
        xoffset = np.random.randint(0, img.shape[1] - logo["logo"].shape[1])
        yoffset = np.random.randint(0, img.shape[0] - logo["logo"].shape[0])
        bg = img.copy()
        name = logo["name"]
        logo = logo["logo"]
        for c in range(3):
            bg[yoffset:yoffset+logo.shape[0], xoffset:xoffset+logo.shape[1], c] = \
                    logo[:,:,c] * (np.divide(logo[:,:,3], 255.0)) + np.multiply(bg[yoffset:yoffset+logo.shape[0], xoffset:xoffset+logo.shape[1], c], (1.0 - np.divide(logo[:,:,3], 255.0)))
        img_path = os.path.join(test_dataset_path, f'image_{i}_{name}.png')
        cv2.imwrite(img_path, bg)

import cv2
import os
import numpy as np
import glob
import pickle
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

DATASET_PATH = "/home/giu/moray/test/data"
OUTPUT_PATH = "v1/models"

images_path = glob.glob(f'{DATASET_PATH}/*.png')
images_path = sorted(images_path, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
all_features = np.zeros((len(images_path), 3))

for i, img_path in enumerate(images_path):
    print(os.path.basename(img_path))
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(img_rgb, axis=(0, 1))
    std_color = np.std(img_rgb, axis=(0, 1))
    all_features[i] = mean_color
    # all_features[i] = np.concatenate([mean_color, std_color])

# print(all_features)

kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
model = Pipeline(steps=[('clf', kmeans)])
model.fit(all_features)
print(model.named_steps['clf'].labels_)

with open(f'{OUTPUT_PATH}/model.pkl', 'wb') as f:
    pickle.dump(model, f)

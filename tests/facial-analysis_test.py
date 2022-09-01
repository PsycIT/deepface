from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import json

img_path = 'dataset/img1.jpg'
img = cv2.imread(img_path)

plt.imshow(img)
plt.imshow(img[:, :, ::-1])

demography = DeepFace.analyze(img_path)
print('demography = ', demography)
print('type of demography = ', type(demography))

# demography_json = json.loads(demography)
# print('demography_json = ', demography_json)

attributes = ['age', 'emotion']
demography2 = DeepFace.analyze(img, attributes)
print('demography2 = ', demography2)
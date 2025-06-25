import os
import matplotlib.pyplot as plt

results_png_path = os.path.join("runs\detect\mammo_yolo_adamw_flip7", "results.png")
img = plt.imread(results_png_path)
plt.figure(figsize=(15,15))
plt.imshow(img)
plt.axis('off')
plt.show()
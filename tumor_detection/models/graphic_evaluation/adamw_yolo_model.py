import os
import matplotlib.pyplot as plt

results_png_path = os.path.join("../mammo_yolo_adamw_flip", "results.png")
img = plt.imread(results_png_path)
plt.figure(figsize=(15,15))
plt.imshow(img)
plt.axis('off')
plt.show()
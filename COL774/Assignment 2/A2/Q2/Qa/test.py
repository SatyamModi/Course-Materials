import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

weight = np.load('weight.npy')
for i in range(len(weight)):
    weight[i][0] = int(255* weight[i][0])
weight = weight.reshape((32, 32, 3))
print(weight.shape)
img = Image.fromarray(weight, 'RGB')
img.save('weight.png')
# img.show()
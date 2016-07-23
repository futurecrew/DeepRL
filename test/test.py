

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fname = '/media/big/download/ic_launcher.png'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
#plt.imshow(arr, cmap='Greys_r')
plt.imshow(arr, cmap='gray')
plt.show()


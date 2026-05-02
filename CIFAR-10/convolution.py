import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('CIFAR-10/chat.jpg')

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

img_gris = 0.299*R + 0.587*G + 0.114*B

K1 = np.array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]) / 9

K2 = np.array([[0, -1, 0],
               [-1, 5, -1],
               [0, -1, 0]])

K3 = np.array([[-1, 2, -1],
               [-1, 2, -1],
               [-1, 2, -1]])

K4 = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
K5 = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
K6 = np.array([[-2, 1, 0],
               [-1, 1, 1],
               [0, 1, 2]])

def convolution(image, K, l=0):
    H, W = image.shape
    image_paddee = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    image_filtree = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = image_paddee[i:i+3,j:j+3]
            image_filtree[i,j]= np.sum(patch*K) + l
    return image_filtree
"""
filtres = [K1, K2, K3, K4, K5, K6]
noms = ['K1 - Flou', 'K2 - Netteté', 'K3 - Bords horizontaux',
        'K4 - Bords verticaux', 'K5 - Sobel vertical', 'K6 - Sobel diagonal']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (K, nom) in enumerate(zip(filtres, noms)):
    img_filtree = convolution(img_gris, K)
    axes[i//3, i%3].imshow(img_filtree, cmap='gray')
    axes[i//3, i%3].set_title(nom)
    axes[i//3, i%3].axis('off')

plt.suptitle('Effets des filtres de convolution')
plt.tight_layout()
plt.show()
"""
def max_pooling(image):
    H, W = image.shape
    nouvelle_image = np.zeros((H//2, W//2))
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            patch = image[i:i+2, j:j+2]
            nouvelle_image[i//2, j//2] = np.max(patch)
    return nouvelle_image

img_filtree = convolution(img_gris, K1)
img_poolee = max_pooling(img_filtree)

print(img_filtree.shape)  # (798, 1058)
print(img_poolee.shape)   # (399, 529)  ← divisé par 2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_filtree, cmap='gray')
plt.title('Après convolution')

plt.subplot(1, 2, 2)
plt.imshow(img_poolee, cmap='gray')
plt.title('Après max-pooling')
plt.show()
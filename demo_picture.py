import numpy as np
import skimage
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
from solve_ncuts import n_cuts


def read_image(filename):
    img = skimage.io.imread(filename)
    if 3 == img.ndim:
        return skimage.img_as_ubyte(skimage.color.rgb2gray(rgb=img))
    else:
        return skimage.img_as_ubyte(img)


def compute_connection(image):
    image_flat = image.ravel()
    n_items = len(image_flat)

    distance = np.zeros((n_items, n_items), dtype=float)
    for which in range(n_items):
        distance[which, :] = np.abs(image_flat - image_flat[which])

    scale_sig = np.max(distance)
    W = np.exp(-np.power(distance / scale_sig, 2))
    return W


def segment_and_show(image, eigenvalues, eigenvectors):
    image_shape = image.shape
    abs_eigenvalues = np.abs(eigenvalues)
    print(abs_eigenvalues[abs_eigenvalues.argsort()])
    # Only show 4 graphs
    ith = 0
    for pos in abs_eigenvalues.argsort()[1:min(5, len(abs_eigenvalues))]:
        eigenvector = eigenvectors[pos] > 0
        cut_area = image * eigenvector.reshape(image_shape)
        # skimage.io.imshow(cut_area)
        skimage.io.imsave('result/result_{}.jpg'.format(ith), cut_area)
        ith += 1


if __name__ == '__main__':
    image = read_image('Ncut_test.jpg')
    W = compute_connection(image)
    (eigenvalues, eigenvectors) = n_cuts(W, regularization=False)
    segment_and_show(image, eigenvalues, eigenvectors)
    input("Press ENTER to continue...")
import cv2
import numpy as np
from numpy import ndarray
import copy

def otsu(img: np.ndarray) -> tuple[int, ndarray]:
    img = copy.deepcopy(img)
    size = img.shape
    all_pixel_num = size[0] * size[1]
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    max_sigma = 0
    optim_th = 0

    omega0 = 0
    cumulative_intensity_class1 = 0

    all_cumulative_intensity = sum([i * hist[i] for i in range(0, 256)])

    for i in range(0, 256):
        omega0 += (hist[i] / all_pixel_num)
        if omega0 == 0:
            continue

        omega1 = 1 - omega0
        if omega1 == 0:
            break

        cumulative_intensity_class1 += i * hist[i]

        mu0 = cumulative_intensity_class1 / (all_pixel_num * omega0)
        mu1 = (all_cumulative_intensity
               - cumulative_intensity_class1) / (all_pixel_num * omega1)

        sigma = omega0 * omega1 * (mu1 - mu0) ** 2

        if sigma >= max_sigma:
            max_sigma = sigma
            optim_th = i

    img[img <= optim_th] = 0
    img[img > optim_th] = 255
    return optim_th, img

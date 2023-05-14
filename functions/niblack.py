import cv2
import numpy as np

def box_filter(img: np.ndarray, ker) -> np.ndarray:
    img = np.pad(img, ((ker + 1, ker), (ker + 1, ker)), mode="reflect")

    integral_img = np.cumsum(img, axis=0)
    integral_img = np.cumsum(integral_img, axis=1)

    integral_image_size = integral_img.shape

    img = (integral_img[0:integral_image_size[0] - 2 * ker - 1, 0:integral_image_size[1] - 2 * ker - 1] +
           integral_img[2 * ker + 1:integral_image_size[0], 2 * ker + 1:integral_image_size[1]] -
           integral_img[0:integral_image_size[0] - 2 * ker - 1, 2 * ker + 1:integral_image_size[1]] -
           integral_img[2 * ker + 1:integral_image_size[0], 0:integral_image_size[1] - 2 * ker - 1])

    img = img / ((2 * ker + 1) ** 2)

    return img.astype(np.uint8)
def niblack(img, core = 1, k=1, a = 0) -> np.ndarray:
    img_size = img.shape
    mu_img = box_filter(img, core)
    mu_img_2 = box_filter(np.power(img,2), core)

    sigma = np.sqrt((mu_img_2 - np.power(mu_img, 2)))

    th = mu_img + k*sigma + a

    img_new = np.zeros(img_size)
    img_new[img > th] = 255
    return img_new

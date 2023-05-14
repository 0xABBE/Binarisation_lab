import numpy as np
from functions.niblack import box_filter

def niblack_multiscale(img, core = 1, sigma_th = 0, k = 1, a = 0) -> np.ndarray:
    img_size = img.shape
    sq_img = img.shape[0]*img.shape[1]

    mu_img = box_filter(img, core)
    mu_img_2 = box_filter(np.power(img,2), core)
    sigma = np.sqrt((mu_img_2 - np.power(mu_img, 2)))
    count = np.count_nonzero(sigma >= sigma_th)

    while count != sq_img and core <= max(img_size):
        core *= 2
        mu_img = box_filter(img, core)
        mu_img_2 = box_filter(np.power(img, 2), core)
        sigma = np.sqrt((mu_img_2 - np.power(mu_img, 2)))
        count = np.count_nonzero(sigma >= sigma_th)

    th = mu_img + k * sigma + a
    img_new = np.zeros(img_size)
    img_new[img > th] = 255

    return img_new

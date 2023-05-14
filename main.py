import numpy as np
import cv2
import argparse
from functions.otsu import otsu
from functions.otsu_noneq import otsu_noneq
from functions.niblack_multiscale import niblack_multiscale
from functions.metrics import f1_score, confusion_matrix
from functions.niblack import niblack


binary = argparse.ArgumentParser()
subparser = binary.add_subparsers(dest="method")

parser_otsu = subparser.add_parser("otsu", help="Starts binarization Otsu")
parser_otsu.add_argument('src', help="Input image's path", type=str)
parser_otsu.add_argument('dst', help="Binary image destination path", type=str)

parser_otsu_noneq = subparser.add_parser("otsunoneq", help="Starts binarization Otsu for noneq classes")
parser_otsu_noneq.add_argument('src', help="Input image's path", type=str)
parser_otsu_noneq.add_argument('dst', help="Binary image destination path", type=str)

parser_niblack = subparser.add_parser("niblack", help="Starts binarization niblack")
parser_niblack.add_argument('src', help="Input image's path", type=str)
parser_niblack.add_argument('dst', help="Binary image destination path", type=str)
parser_niblack.add_argument('core', help="Core of algorithm (for n*n kernel size = (n-1)/2)", type=int)
parser_niblack.add_argument('k', help="k parameter", type=float)
parser_niblack.add_argument('a', help="a parameter", type=float)

parser_niblack_ms = subparser.add_parser("niblackms", help="Starts binarization niblack multiscale")
parser_niblack_ms.add_argument('src', help="Input image's path", type=str)
parser_niblack_ms.add_argument('dst', help="Binary image destination path", type=str)
parser_niblack_ms.add_argument('core', help="Core of algorithm (for n*n kernel size = (n-1)/2)", type=int)
parser_niblack_ms.add_argument('sigma', help="threshold for increasing core size", type=float)
parser_niblack_ms.add_argument('k', help="k parameter", type=float)
parser_niblack_ms.add_argument('a', help="a parameter", type=float)


def is_grate_scale(src_path):
    img = cv2.imread(src_path)
    size = img.shape[0] * img.shape[1]

    r = img[0:, 0:, 0]
    g = img[0:, 0:, 1]
    b = img[0:, 0:, 2]

    eq_pixels_1 = np.count_nonzero(r == g)
    eq_pixels_2 = np.count_nonzero(r == b)
    eq_pixels_3 = np.count_nonzero(b == g)

    if not (eq_pixels_1 == eq_pixels_2 and eq_pixels_2 == eq_pixels_3 and eq_pixels_3 == eq_pixels_1):
        return False, img
    else:
        return True, img[0:, 0:, 0]
def main():
    args1 = vars(binary.parse_args())
    if args1["method"] == "otsu":
        src_path = args1["src"]
        pred, img = is_grate_scale(src_path)
        if pred:
            th, result = otsu(img)
            dst_path = args1["dst"]
            cv2.imwrite(dst_path, result)
        else:
            print("Image is not gray scale")

    elif args1["method"] == "otsunoneq":
        src_path = args1["src"]
        pred, img = is_grate_scale(src_path)
        if pred:
            th, result = otsu_noneq(img)
            dst_path = args1["dst"]
            cv2.imwrite(dst_path, result)
        else:
            print("Image is not gray scale")

    elif args1["method"] == "niblack":
        src_path = args1["src"]
        pred, img = is_grate_scale(src_path)
        if pred:
            core = args1["core"]
            k = args1["k"]
            a = args1["a"]
            result = niblack(img, core, k, a)
            dst_path = args1["dst"]
            cv2.imwrite(dst_path, result)
        else:
            print("Image is not gray scale")
    elif args1["method"] == "niblackms":
        src_path = args1["src"]
        pred, img = is_grate_scale(src_path)
        if pred:
            core = args1["core"]
            sigma = args1["sigma"]
            k = args1["k"]
            a = args1["a"]
            result = niblack_multiscale(img, core, sigma, k, a)
            dst_path = args1["dst"]
            cv2.imwrite(dst_path, result)
        else:
            print("Image is not gray scale")
    else:
        print("unknown or unread command")

if __name__ == '__main__':
    main()


import numpy as np


def f1_score(gt, img):
    positive_gt = gt == 0
    positive_img = img == 0
    negative_gt = gt == 255
    negative_img = img == 255

    true_positive = np.count_nonzero(np.logical_and(positive_gt, positive_img))
    false_positive = np.count_nonzero(np.logical_and(negative_gt, positive_img))
    false_negative = np.count_nonzero(np.logical_and(positive_gt, negative_img))

    f1 = 2 * true_positive / (2*true_positive + false_positive + false_negative)
    return f1


def precision(gt, img):
    positive_gt = gt == 0
    positive_img = img == 0
    negative_gt = gt == 255

    true_positive = np.count_nonzero(np.logical_and(positive_gt, positive_img))
    false_positive =np.count_nonzero(np.logical_and(negative_gt, positive_img))

    prec = true_positive / (true_positive + false_positive)
    return prec


def recall(gt, img):
    positive_gt = gt == 0
    positive_img = img == 0
    negative_img = img == 255

    true_positive = np.count_nonzero(np.logical_and(positive_gt, positive_img))
    false_negative = np.count_nonzero(np.logical_and(positive_gt, negative_img))

    rec = true_positive / (true_positive + false_negative)
    return rec


def confusion_matrix(gt, img):
    positive_gt = gt == 0
    positive_img = img == 0
    negative_gt = gt == 255
    negative_img = img == 255

    true_positive = np.count_nonzero(np.logical_and(positive_gt, positive_img))
    true_negative = np.count_nonzero(np.logical_and(negative_gt, negative_img))
    false_positive = np.count_nonzero(np.logical_and(negative_gt, positive_img))
    false_negative = np.count_nonzero(np.logical_and(positive_gt, negative_img))

    matrix = np.array([[true_negative, false_positive],
                       [false_negative, true_positive]])
    return matrix

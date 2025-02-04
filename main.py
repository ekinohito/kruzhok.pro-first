#!./bin/python3
import sys
import os
import json
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


DEBUG = True
"""
True if program needs debug info False otherwise
"""
DOWNSCALE = 0.8
"""
Coefficient of image downscaling in size-search loop
"""
THRESHOLD = 0.35
"""
Manually found threshold of image's score required to be considered positive
"""


def score_image(img, template):
    """

    :param img: scored image
    :param template: template image
    :return: score of given image
    """
    h, w = template.shape
    result = 0
    found_best = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    img_unedited = img.copy()
    img_show = None
    res_show = None
    while img.shape[0] >= h and img.shape[1] >= w:
        edges = cv.Canny(img, 150, 200)
        res = cv.matchTemplate(edges, template, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        result = max(result, max_val)
        if DEBUG:
            if result == max_val:
                found_best = True
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                img_show = edges.copy()
                res_show = res.copy()
        img = cv.resize(img, tuple(map(lambda x: int(x * DOWNSCALE), img.shape[-2::-1])))
    if DEBUG and found_best and result > 0.35:
        cv.rectangle(img_show, top_left, bottom_right, 255, 2)
        plt.subplot(221), plt.imshow(res_show, cmap='gray')
        plt.title(result)
        plt.subplot(222), plt.imshow(img_show, cmap='gray')
        plt.subplot(223), plt.imshow(img_unedited)
        plt.subplot(224), plt.imshow(cv.Canny(img_unedited, 100, 150), cmap='gray')
        plt.show()
    return result


def score_row(directory_path, template):
    """

    :param directory_path: path to scored directory
    :param template: template image
    :return: list of scores
    """
    result = list()
    row_length = len(os.listdir(directory_path))
    for i, file_name in enumerate(os.listdir(directory_path)):
        print(f'\rDone: {i:4d}/{row_length:4d} = {i / row_length * 100:2.0f}%', end='')
        img = cv.imread(os.path.join(directory_path, file_name))
        result.append(score_image(img, template))
    print(f'\rDone: {row_length:4d}/{row_length:4d} = 100%')
    return result


def print_result(scores, adj):
    """

    :param scores: list of scores
    :param adj: what adjective to use
    :return: nothing
    """
    print(f'''Scores on {adj} images:
    min: {min(scores)};
    max: {max(scores)};
    len: {len(scores)};
    avg: {sum(scores) / len(scores)};''')


def plot_hist():
    """
    plot histogram of positive and negatives scores
    :return: nothings
    """
    with open('result.json', 'r') as file:
        scores = json.load(file)
    plt.hist(scores['fake_scores'], bins=np.linspace(0, 1, 200))
    plt.hist(scores['original_scores'], bins=np.linspace(0, 1, 200))
    plt.plot((THRESHOLD, THRESHOLD), (0, 10))
    plt.show()


def analise_dataset():
    """
    analise whole dataset
    :return: fake scores and original scores
    """
    template = cv.Canny(cv.imread('logo50.png', 0), 50, 200)
    fake_scores = score_row(os.path.join('imgs', 'negative'), template)
    original_scores = score_row(os.path.join('imgs', 'positive'), template)
    with open('result.json', 'w+') as file:
        json.dump({'original_scores': original_scores, 'fake_scores': fake_scores}, file)
    return fake_scores, original_scores


def playground():
    """
    analise, print results and plot histogram
    :return: nothing
    """
    fake_scores, original_scores = analise_dataset()
    print_result(fake_scores, 'negative')
    print_result(original_scores, 'positive')
    plot_hist()


def analise_image(file_name, template):
    """

    :param file_name: image's filename
    :param template: template image
    :return: if image contains template
    """
    img = cv.imread(file_name)
    score = score_image(img, template)
    return score >= THRESHOLD


def test():
    """
    analise dataset, print results table and plot histogram
    :return: nothing
    """
    fake_scores, original_scores = analise_dataset()
    correct_positives, false_positives, correct_negatives, false_negatives = [list() for _ in range(4)]
    for score in original_scores:
        if score >= THRESHOLD:
            correct_positives.append(score)
        else:
            false_negatives.append(score)
    for score in fake_scores:
        if score < THRESHOLD:
            correct_negatives.append(score)
        else:
            false_positives.append(score)
    print(f'correct positives: {len(correct_positives)}')
    print(f'correct negatives: {len(correct_negatives)}')
    print(f'false-positives: {len(false_positives)}')
    print(f'false-negatives: {len(false_negatives)}')
    plot_hist()


def auto_canny(image, sigma=0.5):
    """

    :param image: given image
    :param sigma: sigma
    :return: image of edges
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # make gray
    im = cv.medianBlur(gray, 3)
    v = np.median(im)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(im, lower, upper)
    return edged


def main():
    """
    take first command line argument as filename and print "kruzhok" if it contains logo
    :return:
    """
    template = cv.Canny(cv.imread('logo50.png', 0), 50, 200)
    if analise_image(sys.argv[1], template):
        print('kruzhok')


if __name__ == '__main__':
    main()

#!./bin/python3
import sys
import os
import json
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

DEBUG = False
DOWNSCALE = 0.8
THRESHOLD = 0.35


def score_image(img, template):
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
    result = list()
    row_length = len(os.listdir(directory_path))
    for i, file_name in enumerate(os.listdir(directory_path)):
        print(f'\rDone: {i:4d}/{row_length:4d} = {i / row_length * 100:2.0f}%', end='')
        img = cv.imread(os.path.join(directory_path, file_name))
        result.append(score_image(img, template))
    print(f'\rDone: {row_length:4d}/{row_length:4d} = 100%')
    return result


def print_result(scores, adj):
    print(f'''Scores on {adj} images:
    min: {min(scores)};
    max: {max(scores)};
    len: {len(scores)};
    avg: {sum(scores) / len(scores)};''')


def plot_hist():
    with open('result.json', 'r') as file:
        scores = json.load(file)
    plt.hist(scores['fake_scores'], bins=np.linspace(0, 1, 200))
    plt.hist(scores['original_scores'], bins=np.linspace(0, 1, 200))
    plt.plot((THRESHOLD, THRESHOLD), (0, 10))
    plt.show()


def analise_dataset():
    template = cv.Canny(cv.imread(os.path.join('imgs', 'logo50.png'), 0), 50, 200)
    fake_scores = score_row(os.path.join('imgs', 'negative'), template)
    original_scores = score_row(os.path.join('imgs', 'positive'), template)
    with open('result.json', 'w+') as file:
        json.dump({'original_scores': original_scores, 'fake_scores': fake_scores}, file)
    return fake_scores, original_scores


def playground():
    fake_scores, original_scores = analise_dataset()
    print_result(fake_scores, 'negative')
    print_result(original_scores, 'positive')
    plot_hist()


def analise_image(file_name, template):
    img = cv.imread(file_name)
    score = score_image(img, template)
    return score >= THRESHOLD


def test():
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


def main():
    template = cv.Canny(cv.imread(os.path.join('imgs', 'logo50.png'), 0), 50, 200)
    analise_image(sys.argv[1], template)


if __name__ == '__main__':
    main()

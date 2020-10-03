import cv2 as cv
import os
from matplotlib import pyplot as plt

DEBUG = True
DOWNSCALE = 0.8


def score_image(img, template):
    h, w = template.shape
    result = 0
    found_best = False
    top_left = (0, 0)
    bottom_right = (0, 0)
    img_show = None
    res_show = None
    while img.shape[0] >= h and img.shape[1] >= w:
        edges = cv.Canny(img, 150, 200)
        res = cv.matchTemplate(edges, template, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        result = max(result, max_val)
        if DEBUG:
            print(max_val)
            if result == max_val:
                found_best = True
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                img_show = edges.copy()
                res_show = res.copy()
        img = cv.resize(img, tuple(map(lambda x: int(x * DOWNSCALE), img.shape[::-1])))
    if DEBUG and found_best:
        cv.rectangle(img_show, top_left, bottom_right, 255, 2)
        plt.subplot(121), plt.imshow(res_show, cmap='gray')
        plt.subplot(122), plt.imshow(img_show, cmap='gray')
        plt.show()
    return result


def test(directory_path, template):
    result = list()
    for file_name in os.listdir(directory_path):
        img = cv.imread(os.path.join(directory_path, file_name), 0)
        result.append(score_image(img, template))
    return result


def print_result(scores, adj):
    print(f'''Scores on {adj} images:
    min: {min(scores)};
    max: {max(scores)};
    len: {len(scores)};
    avg: {sum(scores) / len(scores)};''')


def main():
    template = cv.Canny(cv.imread(os.path.join('imgs', 'logo50.png'), 0), 50, 200)
    fake_scores = test(os.path.join('imgs', 'negative'), template)
    original_scores = test(os.path.join('imgs', 'positive'), template)
    print_result(fake_scores, 'negative')
    print_result(original_scores, 'positive')


if __name__ == '__main__':
    main()

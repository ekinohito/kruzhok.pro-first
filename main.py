import cv2 as cv
import os
from matplotlib import pyplot as plt

DEBUG = False


def score_image(img, template):
    downscale = 0.9
    h, w = template.shape
    result = 0
    while img.shape[0] >= h and img.shape[1] >= w:
        edges = cv.Canny(img, 150, 200)
        res = cv.matchTemplate(edges, template, cv.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        result = max(result, max_val)
        if DEBUG:
            print(max_val)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            img_show = edges.copy()
            cv.rectangle(img_show, top_left, bottom_right, 255, 2)
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.subplot(122), plt.imshow(img_show, cmap='gray')
            plt.show()
        img = cv.resize(img, tuple(map(lambda x: int(x * downscale), img.shape[::-1])))
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
    fake_scores = test(os.path.join('imgs', 'fake'), template)
    original_scores = test(os.path.join('imgs', 'original'), template)
    print_result(fake_scores, 'wrong')
    print_result(original_scores, 'right')


if __name__ == '__main__':
    main()

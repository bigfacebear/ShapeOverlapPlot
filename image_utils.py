import math
import cv2
import numpy as np

def rotateImage(input_img, angle):
    """
    Rotate the input_img by angle degrees, Rotate center is image center.
    :param input_img:np.array, the image to be rotated
    :param angle:float, the counterclockwise rotate angle
    :return:np.array, the rotated image
    """
    radian = angle * math.pi / 180.0
    a, b = math.sin(radian), math.cos(radian)
    h, w = input_img.shape
    w_r = int(math.ceil(h * math.fabs(a) + w * math.fabs(b)))
    h_r = int(math.ceil(w * math.fabs(a) + h * math.fabs(b)))
    dx, dy = max(0, (w_r - w) / 2), max(0, (h_r - h) / 2)
    img_rotate = cv2.copyMakeBorder(input_img, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=(0,0,0))
    center = (img_rotate.shape[1] / 2.0, img_rotate.shape[0] / 2.0)
    affine_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotate = cv2.warpAffine(img_rotate, affine_matrix, (img_rotate.shape[1], img_rotate.shape[0]), flags=cv2.INTER_NEAREST)
    return img_rotate


def getOverlapArea(lock, key, pos):
    """
    Calculate the overlap area between a lock image and a key image.
    The center of key image lies on pos in lock image.
    :param lock:np.array, lock image
    :param key:np.array, key image
    :param pos:2D-tuple, position of the center of the key image on lock image
    :return:int the overlap area
    """
    lock_h, lock_w = lock.shape
    key_h, key_w = key.shape
    lock_lt = np.array((max(0, pos[0] - key_w / 2), max(0, pos[1] - key_h / 2)))
    lock_rb = np.array((min(lock_w, pos[0] + key_w / 2 + 1), min(lock_h, pos[1] + key_h / 2 + 1)))
    key_lt = np.array((max(0, key_w / 2 - pos[0]), max(0, key_h / 2 - pos[1])))
    key_rb = np.array((min(key_w, -pos[0] + key_w / 2 + lock_w), min(key_h, -pos[1] + key_h / 2 + lock_h)))
    lock_size, key_size = lock_rb - lock_lt, key_rb - key_lt
    intersect_size = np.minimum(lock_size, key_size)
    if intersect_size[0] <= 0 or intersect_size[1] <= 0:
        return 0
    lock_rb, key_rb = lock_lt + intersect_size, key_lt + intersect_size
    intersect = lock[lock_lt[1]:lock_rb[1], lock_lt[0]:lock_rb[0]] & key[key_lt[1]:key_rb[1], key_lt[0]:key_rb[0]]
    return len(intersect.nonzero()[0])


def overlapArea(L, K, pos, angle):
    return getOverlapArea(L, rotateImage(K, angle), pos)

def getArea(input):
    return len(input.nonzero()[0])


if __name__ == '__main__':
    L = cv2.imread('./58.png', cv2.IMREAD_GRAYSCALE)
    K = cv2.imread('./59.png', cv2.IMREAD_GRAYSCALE)

    print overlapArea(L, K, 45, (L.shape[1]/2, L.shape[0]/2))

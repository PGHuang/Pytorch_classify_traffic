import numpy as np
import random
from sklearn.preprocessing import normalize


def random_lighting(img, prob=0.5, light_range=(50, 80), zone=(50, 100), scale=3.0):
    r'''
    This lighting module is based on https://github.com/hujiulong/blog/issues/2
    which
    :param img: [h, w, c] shape
    :param prob: the probability of lighting augmentation
    :param light_range: the lighting color range for each rgb channel
    :param zone: the lighting projected area
    :param scale: the range of lighting center area.
    :return:
    '''
    # only < prob we need to add lighting
    if random.random() > prob:
        return img

    h, w, c = img.shape
    # TODO: light color for projection, you can change it
    light = [random.randint(light_range[0], light_range[1]) for _ in range(3)]

    # get random light position for lighting
    lcx = random.randint(0, int(w * scale))
    lcy = random.randint(0, int(h * scale))
    # random light zone of lighting, range is bigger when it increase
    rand_zone = random.randint(zone[0], zone[1])
    lightPos = np.array([lcy, lcx, rand_zone]).reshape(1, -1)

    # get image all coord
    yv, xv = np.meshgrid(range(0, h), range(0, w), indexing='ij')
    z = np.zeros((h, w), dtype=np.int32)
    positions = np.stack([yv, xv, z], axis=2)

    # normal vector
    normal = [0, 0, 1]
    # change to int for add op
    img = img.astype(np.int32)
    # add lighting for each pixel
    for hidx in range(h):
        currentLight = normalize(lightPos - positions[hidx])
        # lighting projection vector
        intensity = np.dot(currentLight, normal).reshape(-1, 1)
        # do lighting change
        img[hidx] = (img[hidx] + light) * intensity

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    import cv2
    import glob
    import time

    names = glob.glob('./img/*.jpg')
    for name in names:
        img = cv2.imread(name)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        img_cp = img.copy()

        tic = time.time()
        img = random_lighting(img)
        toc = time.time()
        print('Time used:', toc - tic)
        cv2.imshow('check', img)
        cv2.imshow('orig', img_cp)
        cv2.waitKey(0)

import numpy as np
from skimage import exposure
import random
import cv2

def adjustData(img,mask):
    # img = img / 255

    mask = mask /255
    mask[mask > 0.8] = 1
    mask[mask <= 0.8] = 0

    # for i in range(0, img.shape[0]):
    #     # img_print = img[i, :, :, :]
    #     mask_print = mask[i, :, :, 0]
    #     # plt.imshow(img_print)
    #     # plt.show()
    #     plt.imshow(mask_print)
    #     plt.show()

    return (img,mask)



def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


def unet_preprocess(img):
    img = img / 255
    # img = AHE(img)
    # img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    return img

def rotation(img, mask):
    number = random.randint(1, 4)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 0, 1.0)

    if number == 1:
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
    elif number == 2:
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
    elif number == 3:
        M = cv2.getRotationMatrix2D(center, 270, 1.0)

    img = cv2.warpAffine(img, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    return img, mask


def get_augmented(aug, img, mask):

    if aug == "AHE":
        img = AHE(img)
    elif aug == "AHE_rotation":
        img = AHE(img)
        img, mask = rotation(img, mask)
        mask = np.expand_dims(mask, axis=2)
    elif aug == "rotation":
        img, mask = rotation(img, mask)
        mask = np.expand_dims(mask, axis=2)


    return img, mask
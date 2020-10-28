import numpy as np
from skimage.exposure import equalize_adapthist as CLAHE
import random
import cv2

def mask_to_one_hot(mask, num_classes):
    mask_ret = np.zeros((mask.shape[0], mask.shape[1], num_classes))
    for idx in range(num_classes):
        class_ = (mask == idx)[:,:,0]
        mask_ret[:, :, idx] = np.where(class_, 1, 0)

    return mask_ret


def adjustData(img,mask):
    # img = img / 255

    import matplotlib.pyplot as plt
    # mask = mask /255
    mask = np.where(mask > 0.8, 0, 1)
    # mask = np.where(mask > 0.8, 1, 0)

    #cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/img.bmp", img*255)
    #cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/mask.bmp", mask*255)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(mask)
    # plt.show()

    return (img,mask)



def AHE(img):
    img_adapteq = CLAHE(img, clip_limit=0.03)
    return img_adapteq


def unet_preprocess(img):
    img = img / 255
    img = AHE(img)
    # img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    return img

def flip(img, mask):
    number = random.randint(1, 3)

    if number == 1:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    if number == 2:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if number == 3:
        img = cv2.flip(img, -1)
        mask = cv2.flip(mask, -1)

    return img, mask


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


def get_augmented_test(aug, img):
    if aug == "AHE":
        img = CLAHE(img)
    elif aug == "AHE_rotation":
        img = CLAHE(img)
        img, _ = rotation(img, img)
    elif aug == "AHE_flip":
        img = CLAHE(img)
        img, _ = flip(img, img)
    elif aug == "rotation":
        img, _ = rotation(img, img)
    elif aug == "flip":
        img, _ = flip(img, img)
    elif aug == "AHE_rotation_flip":
        img = CLAHE(img)
        img, _ = rotation(img, img)
        img, _ = flip(img, img)

    return img


def get_augmented(aug, img, mask):
    if aug == "AHE":
        img = CLAHE(img)
    elif aug == "AHE_rotation":
        img = CLAHE(img)
        img, mask = rotation(img, mask)
        mask = np.expand_dims(mask, axis=2)
    elif aug == "AHE_flip":
        img = CLAHE(img)
        img, mask = flip(img, mask)
        mask = np.expand_dims(mask, axis=2)
    elif aug == "rotation":
        img, mask = rotation(img, mask)
        mask = np.expand_dims(mask, axis=2)
    elif aug == "flip":
        img, mask = flip(img, mask)
        mask = np.expand_dims(mask, axis=2)
    elif aug == "AHE_rotation_flip":
        img = CLAHE(img)
        img, mask = rotation(img, mask)
        img, mask = flip(img, mask)
        mask = np.expand_dims(mask, axis=2)

    return img, mask

def result_map_to_img(result):
    img = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)
    res_map = np.squeeze(result)
    img = np.squeeze(img)

    argmax_idx = np.argmax(res_map, axis=2)
    # For np.where calculation.
    background = (argmax_idx == 0)
    # soldering = (argmax_idx == 1)
    # break_ = (argmax_idx == 2)
    crack = (argmax_idx == 1)
    finger = (argmax_idx == 2)
    microcrack = (argmax_idx == 3)

    #blue --> background
    #green --> finger
    #red --> crack, microcrack

    img[:, :, 0] = np.where(background, 255, img[:, :, 0])
    img[:, :, 1] = np.where(background, 255, img[:, :, 1])
    img[:, :, 2] = np.where(background, 255, img[:, :, 2])

    # img[:, :, 1] = np.where(soldering, 128, img[:, :, 1])
    # img[:, :, 1] = np.where(break_, 255, img[:, :, 1])
    img[:, :, 1] = np.where(finger, 255, img[:, :, 1])
    img[:, :, 2] = np.where(crack, 128, img[:, :, 2])
    img[:, :, 2] = np.where(microcrack, 255, img[:, :, 2])



    return img


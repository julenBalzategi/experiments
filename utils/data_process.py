import numpy as np
from skimage import exposure

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
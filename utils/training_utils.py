import glob
import random

import tensorflow as tf
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2
from utils.data_process import adjustData, result_map_to_img, mask_to_one_hot
from utils import data_process
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import os
from keras.models import load_model
import re
from PIL import Image

imgs_type = ["bmp", "jpg", "tiff"]

def get_train_generator(train_generator, batch_size, train_path, num_img, target_size, aug, num_classes, classes=None,
                        folds=None, cell_types=None):

    if folds is None:
        folds = ["1", "2", "3"]
    if train_generator == "custom":
        generator = train_generator_custom(batch_size, train_path, num_img, target_size, aug=aug, folds=folds, cell_types=cell_types)
    elif train_generator == "multiclass":
        generator = train_generator_multiclass(batch_size, train_path, num_img, target_size, aug=aug, num_classes=num_classes,
                                               classes=classes, folds=folds, cell_types=cell_types)
    else:
        generator = trainGenerator(batch_size, train_path, target_size=target_size, aug=aug, folds=folds, cell_types=cell_types)

    return generator

#TODO: refactor to consider new folder order
def trainGenerator(batch_size, train_path, num_class = 2, target_size=(400, 400), aug="None", folds=None, cell_types=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    if folds is None:
        folds = ["1", "2", "3"]
    images_path = []
    masks_path = []
    for fold in folds:
        for cell_type in cell_types:
            for img_type in imgs_type:
                images_path.extend(
                    glob.glob(os.path.join(train_path, "fold" + fold, "img/", cell_type, "*." + img_type)))
                masks_path.extend(
                    glob.glob(os.path.join(train_path, "fold" + fold, "label/", cell_type, "*." + img_type)))
    data_gen_args = dict(rotation_range=0.2)


    image_datagen = ImageDataGenerator(data_gen_args)#, rescale=1.0/255.0)
    mask_datagen = ImageDataGenerator(data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        #classes=["cell"],
        class_mode = None,
        color_mode="rgb",
        target_size = target_size,
        batch_size = batch_size,
        seed = 1,
        follow_links=images_path)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        #classes = ["label"],
        class_mode= None,
        color_mode="grayscale",
        target_size = target_size,
        batch_size = batch_size,
        seed = 1,
        follow_links=masks_path)

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        for idx in range(len(img[:])):
            img[idx],mask[idx] = data_process.get_augmented(aug, img[idx], mask[idx])
        yield (img,mask)

def train_generator_custom(batch_size, train_path, num_img = 1, num_class = 2, target_size=(400, 400), aug="None",
                           folds=None, cell_types=None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    if folds is None:
        folds = ["1", "2", "3"]
    assert batch_size <= num_img, "The batch size must be lower or equal to the num of image"

    data_gen_args = dict(rotation_range=0.2)

    image_datagen = ImageDataGenerator(data_gen_args, rescale=1.0 / 255.0)
    mask_datagen = ImageDataGenerator(data_gen_args)

    images_path = []
    masks_path = []
    for fold in folds:
        for cell_type in cell_types:
            for img_type in imgs_type:
                images_path.extend(glob.glob(os.path.join(train_path, "fold"+fold, "img/", cell_type, "*."+img_type)))
                masks_path.extend(glob.glob(os.path.join(train_path, "fold"+fold, "label/", cell_type, "*."+img_type)))
    # images_path.extend(glob.glob(os.path.join(train_path, "train", "cell", "*.png")))
    # masks_path.extend(glob.glob(os.path.join(train_path, "train", "label", "*.png")))

    ##TODO: temporary fix. check out if these two line affects also to poly dataset.
    images_path.sort()
    masks_path.sort()
    ##
    x = []
    y = []
    generator = list(zip(images_path[:num_img], masks_path[:num_img]))
    # random.shuffle(generator)
    while True:

        train_pair = random.sample(generator, batch_size)

        for idx in range(batch_size):
            img = cv2.imread(train_pair[idx][0])
            # plt.imshow(img)
            # plt.show()
            mask = cv2.imread(train_pair[idx][1], -1)#[:,:,0]
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = np.expand_dims(mask, 2)
            # plt.imshow(mask)
            # plt.show()

            img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

            mask = np.expand_dims(mask, 2)

            x.append(img)
            y.append(mask)

        x_tmp_gen = image_datagen.flow(np.array(x),
                                       batch_size=batch_size,
                                       shuffle=False
                                       # seed=seed)
                                       )
        y_tmp_gen = mask_datagen.flow(np.array(y),
                                      batch_size=batch_size,
                                      shuffle=False
                                      # seed=seed)
                                      )

        # Finally, yield x, y data.
        x_result = next(x_tmp_gen)
        y_result = next(y_tmp_gen)

        for idx in range(batch_size):
            # plt.imshow(x_result[idx])
            # plt.show()
            # plt.imshow(y_result[idx])
            # plt.show()
            x_result[idx], y_result[idx] = adjustData(x_result[idx], y_result[idx])
            x_result[idx], y_result[idx] = data_process.get_augmented(aug, x_result[idx], y_result[idx])
            # plt.imshow(x_result[idx])
            # plt.show()
            # plt.imshow(y_result[idx])
            # plt.show()
        yield x_result, y_result
        # plt.close()

        x.clear()
        y.clear()

def train_generator_multiclass(batch_size, train_path, num_img, target_size, aug, num_classes, classes, folds, cell_types):
    '''
      can generate image and mask at the same time
      use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
      if you want to visualize the results of generator, set save_to_dir = "your path"
      '''
    class_dict = dict(zip(classes, range(len(classes))))
    data_gen_args = dict(rotation_range=0.2)

    image_datagen = ImageDataGenerator(data_gen_args, rescale=1.0 / 255.0)
    mask_datagen = ImageDataGenerator(data_gen_args)
    # image_generator = image_datagen.flow_from_directory(
    #     train_path,
    #     classes=["cell"],
    #     class_mode=None,
    #     color_mode="rgb",
    #     target_size=target_size,
    #     batch_size=batch_size,
    #     seed=1)
    # mask_generator = mask_datagen.flow_from_directory(
    #     train_path,
    #     classes=["label"],
    #     class_mode=None,
    #     color_mode="rgb",
    #     target_size=target_size,
    #     batch_size=batch_size,
    #     seed=1)
    #
    # train_generator = zip(image_generator, mask_generator)
    # for (img, mask) in train_generator:
    #     # print_debug(mask, img)
    #     # mask = mask_to_one_hot(mask, classes)
    #     mask = to_categorical(mask)
    #     # print_debug2(mask, img)
    #     for idx in range(len(img)):
    #         # plt.imshow(img[idx])
    #         # plt.show()
    #         # plt.imshow(mask[idx][:, :, 0])
    #         # plt.show()
    #         img[idx], mask[idx] = data_process.get_augmented(aug, img[idx], mask[idx])
    #
    #     yield (img, mask)

    images_path = []
    masks_path = []
    for fold in folds:
        for cell_type in cell_types:
            for img_type in imgs_type:
                for class_ in classes:
                    images_path.extend(
                        glob.glob(os.path.join(train_path, "fold" + fold, "img/", cell_type, class_, "*." + img_type)))
                    masks_path.extend(
                        glob.glob(os.path.join(train_path, "fold" + fold, "label/", cell_type, class_, "*." + img_type)))

    images_path.sort()
    masks_path.sort()

    x = []
    y = []
    #images_path, masks_path = shuffle(images_path, masks_path)
    #generator = list(zip(images_path, masks_path))
    while True:

        # train_pair = sample_batch(images_path, batch_size, classes[1:])

        for idx in range(batch_size):
            idx_ = np.random.randint(0, len(images_path))
            img = cv2.imread(images_path[idx_])
            mask = cv2.imread(masks_path[idx_], 0)

            class_ = re.findall("(?<=/).*?(?=/)", masks_path[0])[-1]

            mask = np.where(mask == 0, class_dict.get(class_), 0)

            img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

            mask = np.expand_dims(mask, 2)
            mask = mask_to_one_hot(mask, num_classes)
            # mask = to_categorical(mask, num_classes=len(classes))

            x.append(img)
            y.append(mask)

        x_tmp_gen = image_datagen.flow(np.array(x),
                                       batch_size=batch_size,
                                       # shuffle=True,
                                       # seed=seed)
                                       )
        y_tmp_gen = mask_datagen.flow(np.array(y),
                                      batch_size=batch_size,
                                      # shuffle=True,
                                      # seed=seed)
                                      )

        x_result = next(x_tmp_gen)
        y_result = next(y_tmp_gen)

        for idx in range(batch_size):
            x_result[idx], y_result[idx] = data_process.get_augmented(aug, x_result[idx], y_result[idx])

        yield x_result, y_result
        # plt.close()

        x.clear()
        y.clear()

def print_debug(mask, img_):
    import time
    for idx in range(len(mask)):
        img = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

        argmax_idx = mask[idx][:,:,0]
        # For np.where calculation.
        background = (argmax_idx == 0)
        # soldering = (argmax_idx == 1)
        # break_ = (argmax_idx == 2)
        crack = (argmax_idx == 3)
        finger = (argmax_idx == 4)
        microcrack = (argmax_idx == 5)

        img[:, :, 0] = np.where(background, 255, 0)
        # img[:, :, 1] = np.where(soldering, 128, img[:, :, 1])
        # img[:, :, 1] = np.where(break_, 255, img[:, :, 1])
        img[:, :, 1] = np.where(finger, 255, img[:, :, 1])
        img[:, :, 2] = np.where(crack, 128, img[:, :, 2])
        img[:, :, 2] = np.where(microcrack, 255, img[:, :, 2])

        time_ = str(time.time())
        cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/prueba_{}_label.png".format(time_), img)
        cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/prueba_{}_img.png".format(time_), img_[idx]*255)
        # plt.imshow(img)
        # plt.show()

def print_debug2(mask, img_):
    import time
    for idx in range(len(mask)):
        img = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

        argmax_idx = np.argmax(mask[idx], axis=2)
        # For np.where calculation.
        background = (argmax_idx == 0)
        # soldering = (argmax_idx == 1)
        # break_ = (argmax_idx == 2)
        crack = (argmax_idx == 3)
        finger = (argmax_idx == 4)
        microcrack = (argmax_idx == 5)

        img[:, :, 0] = np.where(background, 255, 0)
        # img[:, :, 1] = np.where(soldering, 128, img[:, :, 1])
        # img[:, :, 1] = np.where(break_, 255, img[:, :, 1])
        img[:, :, 1] = np.where(finger, 255, img[:, :, 1])
        img[:, :, 2] = np.where(crack, 128, img[:, :, 2])
        img[:, :, 2] = np.where(microcrack, 255, img[:, :, 2])

        time_ = str(time.time())
        cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/prueba_{}_label_recon.png".format(time_), img)
        cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/prueba_{}_img_reconst.png".format(time_), img_[idx]*255)
        # plt.imshow(img)
        # plt.show()

class TrainCheck(Callback):

    def __init__(self, sheet, test_name, input_size, train_preprocess, poly):
        super().__init__()
        self.epoch = 0
        self.input_size = input_size
        self.test_name = test_name
        self.sheet = sheet
        self.train_preprocess = train_preprocess
        self.poly = poly

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        if self.poly:
            self.visualize("/home/jbalzategi/datasets/dataset_solar/poly_cross/defective/by_size/fold1/img/4_buses/499.bmp", "train")
            self.visualize('/home/jbalzategi/datasets/dataset_solar/poly_cross/defective/by_size/fold3/img/4_buses/464.bmp', "test")
        else:
            self.visualize("/home/jbalzategi/datasets/dataset_solar/mono_cross/defective/random/fold1/img/4_buses/32.jpg", "train")
            self.visualize('/home/jbalzategi/datasets/dataset_solar/mono_cross/defective/random/test/img/4_buses/602067_Image_5_4.jpg', "test")
            # self.visualize(
            #     "/home/jbalzategi/datasets/dataset_solar/endeas_unet/defective/by_size/fold1/img/4_buses/a_11.tiff", "train")
            # self.visualize('/home/jbalzategi/datasets/dataset_solar/endeas_unet/defective/by_size/test/img/4_buses/c_10.tiff', "test")

    def visualize(self, path, set):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = getattr(data_process, self.train_preprocess)(img)
        # cv2.imwrite("/home/jbalzategi/tmp/debug_prueba/img_prueba.png", img[0]*255)

        # self.model.trainable = False
        pred = self.model.predict(img)
        # self.model.trainable = True
        pred = pred[0]*255
        cv2.imwrite("./tests/{}/{}/training_images/img_epoch_{}_{}.png".format(self.sheet, self.test_name, self.epoch, set), pred)

        # writer = tf.summary.FileWriter("./tests/{}/{}/Graph".format(self.sheet, self.test_name))
        # writer.add_summary(summary=tf.Summary.Value(value=pred))

class TrainCheck_mono_multiclass(Callback):

    def __init__(self, sheet, test_name, input_size, train_preprocess, poly):
        super().__init__()
        self.epoch = 0
        self.input_size = input_size
        self.test_name = test_name
        self.sheet = sheet
        self.train_preprocess = train_preprocess

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        # self.visualize("/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/train/img/6_break_img.jpg",
        #                "train_break")
        self.visualize("/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/train/cell/27_crack_img.jpg",
                       "train_crack")
        self.visualize("/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/train/cell/39_finger_img.jpg",
                       "train_finger")
        self.visualize("/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/train/cell/230_micro_img.jpg",
                       "train_micro")
        # self.visualize("/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/train/img/2_badSolder_img.jpg",
        #                "train_badSolder")
        self.visualize('/home/jbalzategi/datasets/dataset_solar/mono/old/train/cell/30.jpg', "test")

        self.visualize('/home/jbalzategi/datasets/dataset_solar/mono/Multilabel/validate/cell/105_finger_img.jpg', "validate")

    def visualize(self, path, set):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = getattr(data_process, self.train_preprocess)(img)

        self.model.trainable = False
        pred = self.model.predict(img)
        self.model.trainable = True
        pred = result_map_to_img(pred)
        cv2.imwrite("./tests/{}/{}/training_images/img_epoch_{}_{}.png".format(self.sheet, self.test_name, self.epoch, set), pred)

def testGenerator(test_path, target_size = (400,400), classes=None, aug=None):
    '''
        can generate image and mask at the same time
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''


    # images_path = []
    # for cell_type in classes:
    #     for img_type in imgs_type:
    #         images_path.extend(glob.glob(os.path.join(test_path, "test", "img/",  cell_type,"*."+img_type)))

    # images_path = []
    # for img_type in imgs_type:
    #     images_path.extend(glob.glob(os.path.join(test_path, "*", "Module/","*."+img_type)))

    ##TMP##
    defective = "/home/jbalzategi/datasets/dataset_solar/validation_anomaly/defective/auto/test/*/img/4_buses"
    defect_free = "/home/jbalzategi/datasets/dataset_solar/validation_anomaly/defect_free/img/*/4_buses"

    images_path = []
    samples = [defective, defect_free]
    for i in samples:
        for img_type in imgs_type:
            images_path.extend(glob.glob(os.path.join(i, "*."+img_type)))
    ####
    # images_path = []
    # for img_type in imgs_type:
    #     images_path.extend(glob.glob(os.path.join(test_path, "*", "img", "4_buses","*."+img_type)))
    ##
    x = []
    def generator():

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            else:
                img = np.full((400,400, 3), fill_value=128)
            img = data_process.get_augmented_test(aug, img)
            img = np.expand_dims(img, 0)
            yield img


    return generator(), len(images_path), images_path


def saveResults(results, filenames, test_name, sheet, poly):
    for idx in range(len(results)):
        name = filenames[idx]
        ####poly
        if poly:
            img = results[idx]*255
            #img = 255 - img
            ####mono
        else:
            img = result_map_to_img(np.expand_dims(results[idx], axis=0))
        img = Image.fromarray((img[:,:,0]).astype(np.uint8))
        # save_name = "./tests/{}/{}/test_results/{}".format(sheet,test_name, name.split("/")[-1])
        save_name = "./tests/{}/{}/test_results/{}".format(sheet,test_name, name.split("/")[-1])
        save_name = save_name.replace("jpg", "tiff")
        img.save(save_name, dpi=(1000, 1000))
        # cv2.imwrite("./tests/{}/{}/test_results/{}".format(sheet,test_name, name.split("/")[-1]), img)

def sample_batch(list_train, batch_size, classes):
    "Check if the list of pairs img-label is class balanced"
    ret = []

    shuffle(classes)

    for idx in range(batch_size):
        class_ = random.sample(classes, 1)[0]
        img_item_list = random.sample([x for x in list_train if class_ in x], 1)[0]
        mask_item_list = img_item_list.replace("jpg", "png")
        mask_item_list = mask_item_list.replace("cell", "label")
        mask_item_list = mask_item_list.replace("img", "label")
        ret.append([img_item_list, mask_item_list])

    return ret


def load_model_(path):
    import losses  as l
    model = load_model(path,
                       custom_objects={"dice_coeff_orig_loss": l.dice_coeff_orig_loss,
                                       "dice_coeff_orig": l.dice_coeff_orig,
                                       "categorical_cross_entropy": l.categorical_cross_entropy,
                                       "categorical_cross_entropy_weighted_loss": l.categorical_cross_entropy_weighted_loss,
                                       "categorical_focal_loss_fixed": l.categorical_focal_loss_fixed,
                                       "iou_nobacground": l.iou_nobacground}
                       )

    return model



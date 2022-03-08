
from numpy.random import choice

from inspect import getmembers, isfunction
from tensorflow.python.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import mlflow

import glob
import random

import tensorflow as tf
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import cv2
from utils.data_process import adjust_data, result_map_to_img, mask_to_one_hot
from utils import data_process
import numpy as np
from sklearn.utils import shuffle
import os
from keras.models import load_model
import re
from PIL import Image
import itertools
import losses

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
    item_combi = [folds, cell_types, imgs_type]
    combinations = list(itertools.product(*item_combi))

    for combi in combinations:
        images_path.extend(
                    glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], "*." + combi[2])))
        masks_path.extend(
                    glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], "*." + combi[2])))
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
        img,mask = adjust_data(img,mask)
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
    item_combi = [folds, cell_types, imgs_type]
    combinations = list(itertools.product(*item_combi))

    for combi in combinations:
        images_path.extend(
            glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], "*." + combi[2])))
        masks_path.extend(
            glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], "*." + combi[2])))

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
            mask = cv2.imread(train_pair[idx][1], -1)#[:,:,0]
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = np.expand_dims(mask, 2)

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
            x_result[idx], y_result[idx] = adjust_data(x_result[idx], y_result[idx])
            x_result[idx], y_result[idx] = data_process.get_augmented(aug, x_result[idx], y_result[idx])
        yield x_result, y_result

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

    images_path = []
    masks_path = []

    item_combi = [folds, cell_types, classes, imgs_type]
    combinations = list(itertools.product(*item_combi))

    for combi in combinations:
        images_path.extend(
            glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], combi[2], "*." + combi[3])))
        masks_path.extend(
            glob.glob(os.path.join(train_path, "fold" + combi[0], "img/", combi[1], combi[2], "*." + combi[3])))

    images_path.sort()
    masks_path.sort()

    x = []
    y = []
    classes_batch = ["micro", "crack", "finger"]
    while True:

        idx = 0
        idx_batch = 0
        while idx != batch_size:
            idx_ = np.random.randint(0, len(images_path))
            img = cv2.imread(images_path[idx_])
            mask = cv2.imread(masks_path[idx_], 0)
            idx += 1
            class_ = re.findall("(?<=/).*?(?=/)", masks_path[idx_])[-1]
            if idx_batch != len(classes_batch): #para que en cada batch haya un sample de cada clase
                if class_ == classes_batch[idx_batch]:
                    idx_batch += 1
                else:
                    idx -= 1
                    continue
            if np.array_equal(np.unique(mask), np.array([255])):
                idx -= 1
                continue

            mask = np.where(mask == 0, class_dict.get(class_), 0)

            img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

            mask = np.expand_dims(mask, 2)
            mask = mask_to_one_hot(mask, num_classes)

            x.append(img)
            y.append(mask)

        x_tmp_gen = image_datagen.flow(np.array(x),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       )
        y_tmp_gen = mask_datagen.flow(np.array(y),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      )

        x_result = next(x_tmp_gen)
        y_result = next(y_tmp_gen)

        for idx in range(batch_size):
            x_result[idx], y_result[idx] = data_process.get_augmented(aug, x_result[idx], y_result[idx])


        yield x_result, y_result

        x.clear()
        y.clear()

class LossRecorder(Callback):
    def __init__(self, loss_name):
        super().__init__()
        self.name = loss_name
        self.step = 1

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric(self.name, logs["loss"], step=self.step)
        self.step += 1
        # mlflow.log_metric("Accuracy", logs.accuracy)

class TrainCheck(Callback):

    def __init__(self, sheet, test_name, input_size, train_preprocess, poly, visualize_images):
        super().__init__()
        self.epoch = 0
        self.input_size = input_size
        self.test_name = test_name
        self.sheet = sheet
        self.train_preprocess = train_preprocess
        self.poly = poly
        self.visualize_images = visualize_images.split(",")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        for file in self.visualize_images:
            img, set = file.split(":")
            self.visualize(img, set)

    def visualize(self, path, set):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = getattr(data_process, self.train_preprocess)(img)

        pred = self.model.predict(img)
        pred = pred[0] * 255
        mlflow.log_image(pred, "img_epoch_{}_{}.png".format(self.epoch, set))


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
        # img = getattr(data_process, self.train_preprocess)(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)

        self.model.trainable = False
        pred = self.model.predict(img)
        self.model.trainable = True
        pred = result_map_to_img(pred)
        cv2.imwrite("./tests/{}/{}/training_images/img_epoch_{}_{}.png".format(self.sheet, self.test_name, self.epoch, set), pred)

def test_generator(test_path, target_size = (400,400), classes=None, aug=None):

    ##TMP##
    # defective = "/home/jbalzategi/datasets/dataset_solar/mono_cross/defective/by_size/*/img/4_buses/"
    defective = "/home/jbalzategi/datasets/dataset_solar/mono_cross/defect_free/img/test/4_buses/" # "/home/jbalzategi/datasets/dataset_solar/validation_anomaly/defect_free/img/fold1/4_buses/"#"/home/jbalzategi/datasets/dataset_solar/validation_anomaly/defective/manual/*/*/img/4_buses"
    # defect_free = "/home/jbalzategi/datasets/dataset_solar/validation_anomaly/defect_free/img/*/4_buses"

    images_path = []
    # samples = [defective, defect_free]
    samples = [defective]
    for i in samples:
        for img_type in imgs_type:
            images_path.extend(glob.glob(os.path.join(i, "*."+img_type)))

    def generator():

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
            else:
                img = np.full((target_size[0],target_size[1], 3), fill_value=128)
            img, _ = data_process.get_augmented(aug, img, img)
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
        save_name = "./tests/{}/{}/test_results/{}".format(sheet,test_name, name.split("/")[-1])
        save_name = save_name.replace("jpg", "tiff")
        img.save(save_name, dpi=(1000, 1000))

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
    if ".h5" in path:
        all_losses = {o[0]:o[1] for o in getmembers(losses) if isfunction(o[1])}
        model = load_model(path,
                       custom_objects=all_losses
                       )
    else:
        model = mlflow.keras.load_model(path)
    
    # for i in range(len(model.layers[:-1])):
    #     model.layers[i].trainable = False
    return model


def get_optimizer(optimizer, lr, decay):
    #This way of importing optimizers is due to how Tensorflow encapsulates the functions

    if optimizer == "Adam":
        return Adam(learning_rate=lr, decay=decay)
    if optimizer == "RMSprop":
        return RMSprop(learning_rate=lr, decay=decay)

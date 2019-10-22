from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import cv2
from utils.data_process import adjustData
from utils import data_process
import matplotlib.pyplot as plt
import numpy as np


def get_train_generator(train_generator, batch_size, train_path, num_img, target_size, aug):

    if train_generator == "custom":
        generator = train_generator_custom(batch_size, train_path, num_img, target_size, aug=aug)
    else:
        generator = trainGenerator(batch_size, train_path, target_size=target_size, aug=aug)

    return generator

def trainGenerator(batch_size,train_path, num_class = 2, target_size=(400, 400), aug="None"):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    data_gen_args = dict(rotation_range=0.2)


    image_datagen = ImageDataGenerator(data_gen_args, rescale=1.0/255.0)
    mask_datagen = ImageDataGenerator(data_gen_args)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=["cell"],
        class_mode = None,
        color_mode="rgb",
        target_size = target_size,
        batch_size = batch_size,
        seed = 1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=["label"],
        class_mode = None,
        color_mode="grayscale",
        target_size = target_size,
        batch_size = batch_size,
        seed = 1)

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        for idx in range(len(img[:])):
            img[idx],mask[idx] = data_process.get_augmented(aug, img[idx], mask[idx])
        yield (img,mask)

def train_generator_custom(batch_size, train_path, num_img = 1, num_class = 2, target_size=(400, 400), aug="None"):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    assert batch_size <= num_img, "The batch size must be lower or equal to the num of image"

    data_gen_args = dict(rotation_range=0.2)

    image_datagen = ImageDataGenerator(data_gen_args, rescale=1.0 / 255.0)
    mask_datagen = ImageDataGenerator(data_gen_args)

    import glob
    import os
    import random

    images_path = glob.glob(os.path.join(train_path, "cell/*"))
    masks_path = glob.glob(os.path.join(train_path, "label/*"))

    x = []
    y = []
    generator = list(zip(images_path[:num_img], masks_path[:num_img]))
    # random.shuffle(generator)
    while True:

        train_pair = random.sample(generator, batch_size)

        for idx in range(batch_size):
            img = cv2.imread(train_pair[idx][0])
            mask = cv2.imread(train_pair[idx][1], 0)

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


class TrainCheck(Callback):

    def __init__(self, sheet, test_name, input_size, train_preprocess):
        super().__init__()
        self.epoch = 0
        self.input_size = input_size
        self.test_name = test_name
        self.sheet = sheet
        self.train_preprocess = train_preprocess

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch+1
        self.visualize("/home/jbalzategi/datasets/dataset_solar/poly/Luka_version_train_only_defective/train/cell/476.bmp", "train")
        self.visualize('/home/jbalzategi/datasets/dataset_solar/poly/Luka_version/Validation/julen_organization/cell/534.bmp', "test")

    def visualize(self, path, set):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = getattr(data_process, self.train_preprocess)(img)

        pred = self.model.predict(img)
        pred = pred[0]*255
        cv2.imwrite("./tests/{}/{}/training_images/img_epoch_{}_{}.png".format(self.sheet, self.test_name, self.epoch, set), pred)


def testGenerator(test_path, target_size = (400,400)):
    data_gen_args = dict()#preprocessing_function=data_process.AHE)
    test_datagen = ImageDataGenerator(data_gen_args, rescale=1.0/255.0)

    test_generator = test_datagen.flow_from_directory(
                        test_path,
                        classes=["cell"],
                        target_size=target_size,
                        batch_size=1,
                        shuffle=False)

    return test_generator, len(test_generator.filenames), test_generator.filenames

def saveResults(results, filenames, test_name, sheet):
    for idx in range(len(results)):
        name = filenames[idx]
        img = results[idx]*255
        img = 255 - img
        # plt.imshow(np.stack([img[:,:,0], img[:,:,0], img[:,:,0]], axis=2)
        # plt.show()
        cv2.imwrite("./tests/{}/{}/test_results/{}".format(sheet,test_name, name.split("/")[1]), img)

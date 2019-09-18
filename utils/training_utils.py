from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import cv2
from utils.data_process import adjustData
from utils import data_process



def trainGenerator(batch_size,train_path, num_class = 2, target_size=(400, 400)):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    data_gen_args = dict(rotation_range=0.2,
                         horizontal_flip=True,
                         # vertical_flip=True, # wasn't before
                         )#preprocessing_function = AHE)


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
        yield (img,mask)


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
        self.visualize("/home/user/datasets/dataset_solar/poly/Luka_version/Train/julen_organization/cell/6.bmp", "train")
        self.visualize('/home/user/datasets/dataset_solar/poly/Luka_version/Validation/julen_organization/cell/500.bmp', "test")
        # self.visualize('datasets/unet_version_mono/train/cell/1174.jpg', "train")
        # self.visualize('datasets/unet_version_mono/test/cell/6.jpg', "test")

    def visualize(self, path, set):

        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = getattr(data_process, self.train_preprocess)(img)

        pred = self.model.predict(img)
        pred = pred[0]*255
        cv2.imwrite("./tests/{}/{}/training_images/img_epoch_{}_{}.png".format(self.sheet, self.test_name, self.epoch, set), pred)


def testGenerator(test_path, target_size = (400,400)):
    data_gen_args = dict()#preprocessing_function=AHE)
    test_datagen = ImageDataGenerator(data_gen_args, rescale=1.0/255.0)

    test_generator = test_datagen.flow_from_directory(
                        test_path,
                        classes=["cell"],
                        target_size=target_size,
                        batch_size=1,
                        shuffle=False)

    return test_generator, len(test_generator.filenames), test_generator.filenames

def saveResults(results, filenames, test_name):
    for idx in range(len(results)):
        name = filenames[idx]
        img = results[idx]*255
        img = 255 - img
        cv2.imwrite("./tests/{}/test_results/{}".format(test_name, name.split("/")[1]), img)

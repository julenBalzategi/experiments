
from keras.models import load_model

from tqdm import tqdm
import cv2

from PIL import Image
from losses import *
from utils.training_utils import test_generator, result_map_to_img
from utils.excel_reader import ExcelReader
from utils.data_process import get_augmented
import models

test_dataset = "/home/jbalzategi/datasets/dataset_solar/mono_cross/defective/by_size"
sheet = "cross_validation_mono"
reader = ExcelReader("./excel/Libro3.xlsx", sheet)


for test in reader:
    poly = True if "poly" in sheet or "multiclass" not in test.train_dataset else False

    ##LOAD MODEL###################
    model = load_model("./excel/{}/{}/{}.h5".format(sheet, test.name, test.name),
                       custom_objects={"dice_coeff_orig_loss":dice_coeff_orig_loss,
                                       "dice_coeff_orig":dice_coeff_orig,
                                       "categorical_cross_entropy":categorical_cross_entropy,
                                       "categorical_cross_entropy_weighted_loss":categorical_cross_entropy_weighted_loss,
                                       "categorical_focal_loss_fixed":categorical_focal_loss_fixed,
                                       "weighted_dice_coef_loss":weighted_dice_coef_loss,
                                       "iou_nobacground":iou_nobacground,
                                       })#"mse_loss":mse_loss})

    # model.trainable = False
    # model.use_learning_phase = False
    # model.summary()

    #model.load_weights("./tests/{}/{}/{}_model.h5".format(sheet, test.name, test.name))
    testGene, steps, filenames = test_generator(test_dataset, target_size=(test.input_size, test.input_size), classes=test.cell_types.split(","), aug=test.aug_test)

    results = []
    for step in tqdm(range(steps)):

        img = cv2.imread(filenames[step])#, 0)
        if img is not None:
            img = cv2.resize(img, dsize=(test.input_size, test.input_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = np.full((test.input_size, test.input_size, 3), fill_value=128)

        img, _ = get_augmented(aug=test.aug_test, img=img, mask=img)
        img = np.expand_dims(img, 0)

        if "multiclass" in sheet:
            pred = model.predict(img)
            pred = result_map_to_img(pred)
            img = Image.fromarray(pred.astype(np.uint8))
        else:
            #mono orig
            pred = model.predict(img)[0]
            pred = pred * 255
            img = Image.fromarray(pred[:,:,0].astype(np.uint8))

        save_name = f"./excel/{sheet}/{test.name}/test_results/{filenames[step][-58:].replace('/', '_')}"
        save_name = save_name.replace("jpg", "tiff")
        img.save(save_name)

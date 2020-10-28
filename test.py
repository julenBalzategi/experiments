#from tensorflow.keras.models import load_model, Model
from keras.models import load_model, Model
from keras.utils import multi_gpu_model
import glob
from keras import backend as k
import os

# from tqdm import tqdm
import cv2

from PIL import Image
import losses
from losses import *
from utils.training_utils import testGenerator, saveResults, result_map_to_img
from utils.excel_reader import Excel_reader
from utils.data_process import get_augmented_test
import models

#test_dataset = "/home/jbalzategi/sii2020/"
#test_dataset = "/home/jbalzategi/datasets/dataset_solar/poly/Luka_version/Validation/julen_organization/"
# test_dataset = "/home/jbalzategi/neuronel/neuronel_unet/new/datasets/poly/Luka_version/Validation/julen_organization_old/"
# test_dataset = "/home/jbalzategi/datasets/dataset_solar/endeas_unet/defective/by_size"
# test_dataset = "/home/jbalzategi/datasets/dataset_solar/mono_cross_2/defective/by_size"
# test_dataset = "/home/jbalzategi/datasets/Set413-Modulo"
test_dataset = "/home/jbalzategi/datasets/dataset_solar/validation_anomaly/"
sheet = "validacion_anomaly"
reader = Excel_reader("./tests/Libro3.xlsx", sheet)


for test in reader:
    poly = True if "poly" in sheet or "multiclass" not in test.train_dataset else False

    ##LOAD MODEL###################
    model = load_model("./tests/{}/{}/{}.h5".format(sheet, test.name, test.name),
                       custom_objects={"dice_coeff_orig_loss":dice_coeff_orig_loss,
                                       "dice_coeff_orig":dice_coeff_orig,
                                       "categorical_cross_entropy":categorical_cross_entropy,
                                       "categorical_cross_entropy_weighted_loss":categorical_cross_entropy_weighted_loss,
                                       "categorical_focal_loss_fixed":categorical_focal_loss_fixed,
                                       "iou_nobacground":iou_nobacground})

    # model.trainable = False
    # model.use_learning_phase = False
    # model.summary()

    #model.load_weights("./tests/{}/{}/{}_model.h5".format(sheet, test.name, test.name))
    testGene, steps, filenames = testGenerator(test_dataset, target_size=(test.input_size, test.input_size), classes=test.cell_types.split(","), aug=test.aug_test)

    # intermediate_model = Model(model.layers[3].inputs, model.layers[3].get_layer("block4_conv2").output)
    results = []
    for step in range(steps):
        # img = next(testGene)

        img = cv2.imread(filenames[step])
        if img is not None:
            img = cv2.resize(img, dsize=(400, 400), interpolation=cv2.INTER_NEAREST)
        else:
            img = np.full((400, 400, 3), fill_value=128)

        img = get_augmented_test(aug=test.aug_test, img=img)
        img = np.expand_dims(img, 0)

        ##EXECUTE TEST###############
        # results.append(model.predict(img)[0])
        # grads = k.gradients(model.layers[3].get_layer("conv2d_16").output, model.layers[3].get_layer("block4_conv2").output)[0]
        # grads_pooled = k.mean(grads, axis=(0,1,2))
        # iterate = k.function([model.layers[3].inputs], [grads_pooled, model.layers[3].get_layer("block4_conv2").output[0]])
        # pooled_grads_value, conv_layer = iterate([img])
        # for i in range(512):
        #     conv_layer[:,:,i] *=pooled_grads_value[i]
        # heatmap = np.mean(conv_layer, axis=-1)
        # heatmap = np.maximum(heatmap, 0)
        # heatmap /= np.max(heatmap)
        # import matplotlib.pyplot as plt
        # plt.imshow(heatmap)
        # plt.show()
        # if not os.path.exists(filenames[step][:-20]+"/results_multi_class_unet_"):
        #     os.mkdir(filenames[step][:-20]+"/results_multi_class_unet_")
        # save_path = filenames[step].replace("Module", "results_multi_class_unet_")
        if "multiclass" in sheet:
            img = model.predict(img)
            img = result_map_to_img(img)
            img = Image.fromarray(img.astype(np.uint8))
        else:
            img = model.predict(img)[0]
            img = img * 255
            img = Image.fromarray(img[:,:,0].astype(np.uint8))
        # save_path = save_path.replace("jpg", "tiff")
        save_name = "./tests/{}/{}/test_results/{}".format(sheet, test.name, filenames[step][-58:].replace("/", "_"))
        save_name = save_name.replace("jpg", "tiff")
        img.save(save_name)#, dpi=(1000, 1000))
        # img.save(save_path)#, dpi=(1000, 1000))


    ##SAVE RESULTS###############
    # saveResults(results, filenames, test.name, sheet, poly)
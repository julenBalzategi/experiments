from keras.models import load_model
import glob

from losses import dice_coeff_orig_loss, dice_coeff_orig, tversky_loss
from utils.training_utils import testGenerator, saveResults
from utils.excel_reader import Excel_reader
import models

test_dataset = "/home/jbalzategi/datasets/dataset_solar/poly/Luka_version/Validation/julen_organization/"
# test_dataset = "/home/jbalzategi/neuronel/neuronel_unet/new/datasets/poly/Luka_version/Validation/julen_organization_old/"

sheet = "Hoja1"
reader = Excel_reader("./tests/Libro3.xlsx", sheet)

for test in reader:

    ##LOAD MODEL###################
    model = load_model("./tests/{}/{}/{}.h5".format(sheet, test.name, test.name),
                       custom_objects={"dice_coeff_orig_loss":dice_coeff_orig_loss,
                                       "dice_coeff_orig":dice_coeff_orig})

    testGene, steps, filenames = testGenerator(test_dataset)

    ##EXECUTE TEST###############
    results = model.predict_generator(testGene, steps, verbose=1)

    ##SAVE RESULTS###############
    saveResults(results, filenames, test.name, sheet)
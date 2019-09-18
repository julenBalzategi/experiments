from keras.models import load_model
import glob

from losses import dice_coeff_orig_loss, dice_coeff_orig, tversky_loss
from utils.training_utils import testGenerator, saveResults
from utils.excel_reader import Excel_reader

test_dataset = "/home/user/datasets/dataset_solar/poly/Luka_version/Validation/julen_organization/"

reader = Excel_reader("./tests/Libro3.xlsx")

for test in reader:

    ##LOAD MODEL###################
    model_name = glob.glob("./tests/{}/*.h5".format(test.name))
    model = load_model(model_name,
                       custom_objects={"dice_coeff_orig_loss":dice_coeff_orig_loss,
                                       "dice_coeff_orig":dice_coeff_orig,
                                       "tversky_loss": tversky_loss})

    testGene, steps, filenames = testGenerator(test_dataset)

    ##EXECUTE TEST###############
    results = model.predict_generator(testGene, steps, verbose=1)

    ##SAVE RESULTS###############
    saveResults(results, filenames, test.name)
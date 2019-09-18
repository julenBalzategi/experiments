from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.training_utils import trainGenerator, TrainCheck
import keras.optimizers as optim
import json

from utils.excel_reader import Excel_reader
from utils.folder_creator import create_test_folder
import models
import losses

train_dataset = "/home/user/datasets/dataset_solar/poly/Luka_version/Train/julen_organization/"

sheet = "Hoja1"
reader = Excel_reader("./tests/Libro3.xlsx", sheet)


for test in reader:


    create_test_folder(test.name, sheet)

    ##INITIALIZE CALLBACKS########

    checkpoint = ModelCheckpoint(
            filepath="./tests/{}/{}/checkpoint_unet_orig_new_dataset.h5".format(sheet, test.name),
            # monitor='val_' + test_idx['metric'].__name__, #the variable to monitor and save the state with the best value
            monitor=test.metrics,
            save_best_only=True,
            save_weights_only=True)

    earlyStopping = EarlyStopping(monitor="loss", patience=30)

    train_check = TrainCheck(sheet, test.name, test.input_size, test.train_preprocess)

    ##INITIALIZE MODEL############
    model = getattr(models, test.model)(input_size=(test.input_size, test.input_size,3))

    model.compile(optimizer=getattr(optim, test.optim)(lr=test.lr),
                  loss=getattr(losses, test.loss),
                  metrics=[getattr(losses, test.metrics)])

    model.summary()


    generator = trainGenerator(batch_size=test.batch,
                               train_path=train_dataset,
                               target_size=(test.input_size, test.input_size))

    ##TRAIN THE MODEL#######################
    history = model.fit_generator(
            generator=generator,
            steps_per_epoch=test.steps,
            callbacks=[checkpoint, train_check],#, plot_callback], #, earlyStopping],
            epochs=test.epoch,
            verbose=1)

    ##SAVE MODEL###################
    model.save("./tests/{}/{}/unet_after_train_model_poly_new_dataset_pretrained.h5".format(sheet, test.name))
    history_dict = history.history
    json.dump(history_dict, open("./tests/{}/{}/json_history_orig".format(sheet, test.name), 'w'))





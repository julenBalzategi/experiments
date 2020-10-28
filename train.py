
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from utils.training_utils import get_train_generator, TrainCheck, load_model_
import keras.optimizers as optim
import json
from keras.utils import multi_gpu_model
import keras.backend as K
from tensorflow.python import debug as tf_debug
import tensorflow as tf

from utils.excel_reader import Excel_reader
from utils.folder_creator import create_test_folder
from utils import training_utils
import models
import losses

sheet = "validacion_anomaly"
reader = Excel_reader("./tests/Libro3.xlsx", sheet)

poly = True if "poly" in sheet else False

# print(tf.keras.backend.get_session().list_devices())

for test in reader:

    train_dataset = test.train_dataset

    create_test_folder(test.name, sheet)

    ##INITIALIZE CALLBACKS########

    # K.set_session(
    #     tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "a4deb20f1415:6007"))

    checkpoint = ModelCheckpoint(
            filepath="./tests/{}/{}/{}_checkpoint.h5".format(sheet, test.name, test.name),
            # monitor='val_' + test_idx['metric'].__name__, #the variable to monitor and save the state with the best value
            monitor=test.metrics,
            save_best_only=True,
            save_weights_only=True)

    earlyStopping = EarlyStopping(monitor="loss", patience=30)

    train_check = getattr(training_utils, test.train_check)(sheet, test.name, test.input_size, test.train_preprocess, poly=poly)

    TB_callback = TensorBoard(log_dir="./tests/{}/{}/Graph".format(sheet, test.name),
                              write_grads=True,
                              write_images=True,
                              # histogram_freq=1,
                              batch_size=4,
                              write_graph=True)

    ##INITIALIZE MODEL###########
    model = getattr(models, test.model)(input_size=(test.input_size, test.input_size, 3),
                                        pretrained_weights=test.pretrained_weights,
                                        final_activation=test.final_activation,
                                        classes=int(test.num_classes))

    model.summary()
    gpu = test.gpu.split(",")
    if len(gpu) > 1:
        model = multi_gpu_model(model, gpus=len(gpu))

    if test.load_model != "N":
        model = load_model_(test.load_model)

    #todo fix the weighted version of loss and parameter passing
    model.compile(optimizer=getattr(optim, test.optim)(lr=float(test.lr), decay=float(test.decay)),
                  loss=getattr(losses, test.loss),#(weights=[float(i) for i in test.weights.split(",")]),
                  metrics=[getattr(losses, test.metrics), "accuracy"])



    generator = get_train_generator(train_generator=test.train_generator,
                                batch_size=test.batch,
                                train_path=train_dataset,
                                num_img=test.num_img,
                                target_size=(test.input_size, test.input_size),
                                aug=test.aug,
                                num_classes=int(test.num_classes),
                                classes=test.classes.split(","),
                                folds = test.folds.split(","),
                                cell_types = str(test.cell_types).split(","))

    generator_val = get_train_generator(train_generator=test.train_generator,
                                batch_size=test.batch,
                                train_path=train_dataset,
                                num_img=test.num_img,
                                target_size=(test.input_size, test.input_size),
                                aug=test.aug,
                                num_classes=int(test.num_classes),
                                classes=test.classes.split(","),
                                folds = str(test.folds_val).split(","),
                                cell_types = str(test.cell_types).split(","))


    ##TRAIN THE MODEL#######################
    history = model.fit_generator(
            generator=generator,
            validation_data=generator_val,
            validation_steps=test.val_steps,
            steps_per_epoch=test.steps,
            callbacks=[checkpoint, train_check, TB_callback],#, plot_callback], #, earlyStopping],
            epochs=test.epoch,
            verbose=1)

    ##SAVE MODEL###################
    model.save("./tests/{}/{}/{}.h5".format(sheet, test.name, test.name))

    model_json = model.to_json()
    with open("./tests/{}/{}/{}_model_arch.json".format(sheet, test.name, test.name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./tests/{}/{}/{}_model_weights.h5".format(sheet, test.name, test.name))

    #history_dict = history.history
    #json.dump(history_dict, open("./tests/{}/{}/{}_json_history_orig.json".format(sheet, test.name, test.name), 'w'))





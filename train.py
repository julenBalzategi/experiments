import mlflow

from tensorflow.keras.callbacks import ModelCheckpoint
from utils.training_utils import get_train_generator, TrainCheck, load_model_
import tensorflow.keras.optimizers as optim
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from utils.excel_reader import ExcelReader
from utils import training_utils
import models
import losses
from losses import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)


sheet = "cross_validation_mono"
reader = ExcelReader("./excel/Libro3.xlsx", sheet)

poly = True if "poly" in sheet else False

mlflow.set_tracking_uri(f"file:/home/jbalzategi/experiments/tests_runs/mlruns/")
mlflow.set_registry_uri("127.0.0.1:8080")

experiment = mlflow.get_experiment_by_name(f"{sheet}")
if experiment is None:
    experiment = mlflow.create_experiment(f"{sheet}")



for test in reader:
    # mlflow.keras.autolog()
    with mlflow.start_run(run_name=f"{test.name}", experiment_id=f"{experiment.experiment_id}") as run:

        mlflow.log_param("dataset", test.train_dataset)
        mlflow.log_param("preprocess", test.train_preprocess)
        mlflow.log_param("input size", test.input_size)
        mlflow.log_param("loss", test.loss)
        mlflow.log_param("optimizer", test.optim)
        mlflow.log_param("gpus", test.gpu)
        mlflow.log_param("batch", test.batch)

        train_dataset = test.train_dataset

        ##INITIALIZE CALLBACKS########

        checkpoint = ModelCheckpoint(
            filepath="./excel/{}/{}/{}_checkpoint.h5".format(sheet, test.name, test.name),
            # monitor='val_' + test_idx['metric'].__name__, #the variable to monitor and save the state with the best value
            monitor=test.metrics,
            save_best_only=True,
            save_weights_only=True)

        train_check = getattr(training_utils, test.train_check)(sheet, test.name, test.input_size, test.train_preprocess,
                                                                poly=poly, visualize_images=test.visualize_images)

        loss_recorder = training_utils.LossRecorder()

        ##INITIALIZE MODEL###########
        model = getattr(models, test.model)(input_size=(test.input_size, test.input_size, 3),  # 1
                                            pretrained_weights=test.pretrained_weights,
                                            final_activation=test.final_activation,
                                            classes=int(test.num_classes))

        summary = []
        model.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        with open("model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("model_summary.txt")

        mlflow.keras.log_model(model, "models", custom_objects={"dice_coeff_orig_loss": dice_coeff_orig_loss,
                                                            "dice_coeff_orig": dice_coeff_orig,
                                                            "categorical_cross_entropy": categorical_cross_entropy,
                                                            "categorical_cross_entropy_weighted_loss": categorical_cross_entropy_weighted_loss,
                                                            "categorical_focal_loss_fixed": categorical_focal_loss_fixed,
                                                            "iou_nobacground": iou_nobacground})
        gpu = test.gpu.split(",")
        if len(gpu) > 1:
            model = multi_gpu_model(model, gpus=len(gpu))

        if test.load_model != "N":
            model = load_model_(test.load_model)
        else:
            #todo save model
            #just because a bug in mlflow
            #model.save("./model.h5")
            #model = load_model_('./model.h5', compile=False)
            pass

        # todo fix the weighted version of loss and parameter passing
        model.compile(optimizer=getattr(optim, test.optim)(lr=float(test.lr), decay=float(test.decay)),
                      loss=getattr(losses, test.loss),
                      metrics=["accuracy"])

        generator = get_train_generator(train_generator=test.train_generator,
                                        batch_size=test.batch,
                                        train_path=train_dataset,
                                        num_img=test.num_img,
                                        target_size=(test.input_size, test.input_size),
                                        aug=test.aug,
                                        num_classes=int(test.num_classes),
                                        classes=test.classes.split(","),
                                        folds=test.folds.split(","),
                                        cell_types=str(test.cell_types).split(","))

        generator_val = get_train_generator(train_generator=test.train_generator,
                                            batch_size=test.batch,
                                            train_path=train_dataset,
                                            num_img=test.num_img,
                                            target_size=(test.input_size, test.input_size),
                                            aug=test.aug,
                                            num_classes=int(test.num_classes),
                                            classes=test.classes.split(","),
                                            folds=str(test.folds_val).split(","),
                                            cell_types=str(test.cell_types).split(","))

        ##TRAIN THE MODEL#######################
        history = model.fit_generator(
            generator=generator,
            validation_data=generator_val,
            validation_steps=test.val_steps,
            steps_per_epoch=test.steps,
            callbacks=[checkpoint, train_check, loss_recorder],
            epochs=test.epoch,
            verbose=1)

        ##SAVE MODEL###################
        # model.save("./tests/{}/{}/{}.h5".format(sheet, test.name, test.name))
#        mlflow.keras.save_model(model, f"/home/jbalzategi/experiments/tests_runs/{sheet}/{test.name}")

        # model_json = model.to_json()
        # with open("./tests_runs/{}/{}/{}_model_arch.json".format(sheet, test.name, test.name), "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("./tests_runs/{}/{}/{}_model_weights.h5".format(sheet, test.name, test.name))

        # history_dict = history.history
        # json.dump(history_dict, open("./tests/{}/{}/{}_json_history_orig.json".format(sheet, test.name, test.name), 'w'))

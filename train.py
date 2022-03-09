import mlflow
from mlflow.tracking import MlflowClient
from inspect import getmembers, isfunction
from utils.folder_creator import create_experiment_folder
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.training_utils import get_train_generator, TrainCheck, load_model_
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

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

mlflow.set_tracking_uri("http://127.0.0.1:8080")


exp_info = MlflowClient().get_experiment_by_name(f"{sheet}")
exp_id = exp_info.experiment_id if exp_info else MlflowClient().create_experiment(f"{sheet}")




for test in reader:
    # mlflow.keras.autolog()
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{sheet}_{test.name}") as run:
        run_num = run.run_id = run.info.run_uuid

        create_experiment_folder(test.name, sheet)

        model_uri = f"runs:/{run_num}/{sheet}_{test.name}"

        mlflow.log_param("dataset", test.train_dataset)
        mlflow.log_param("preprocess", test.train_preprocess)
        mlflow.log_param("input size", test.input_size)
        mlflow.log_param("loss", test.loss)
        mlflow.log_param("optimizer", test.optim)
        mlflow.log_param("gpus", test.gpu)
        mlflow.log_param("batch", test.batch)

        train_dataset = test.train_dataset

        ##INITIALIZE CALLBACKS########

        earlyStopping = EarlyStopping(monitor="loss", patience=30)

        train_check = getattr(training_utils, test.train_check)(sheet, test.name, test.input_size, test.train_preprocess,
                                                                poly=poly, visualize_images=test.visualize_images)

        loss_recorder = training_utils.LossRecorder(test.loss)

        TB_callback = TensorBoard(log_dir="./models/{}_{}".format(sheet, test.name),
                                  write_grads=True,
                                  write_images=True,
                                  # histogram_freq=1,
                                  batch_size=4,
                                  write_graph=True)

        ##INITIALIZE MODEL###########
        model = getattr(models, test.model)(input_size=(test.input_size, test.input_size, 3),  # 1
                                            pretrained_weights=test.pretrained_weights,
                                            final_activation=test.final_activation,
                                            classes=int(test.num_classes))

        summary = []
        model.summary(print_fn=summary.append)
        summary = "\n".join(summary)
        with open(f"./tests_runs/{sheet}/{test.name}/model_summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact(f"./tests_runs/{sheet}/{test.name}/model_summary.txt")

        all_losses = {o[0]:o[1] for o in getmembers(losses) if isfunction(o[1])}
        mlflow.keras.log_model(model, "models", custom_objects=all_losses)

        gpu = test.gpu.split(",")
        if len(gpu) > 1:
            model = multi_gpu_model(model, gpus=len(gpu))

        if test.load_model != "N":
            model = load_model_(test.load_model)
        else:
            #todo save model
            pass
            #just because a bug in mlflow
            # model.save(f"./tests_runs/{sheet}/{test.name}/{sheet}_{test.name}_model.h5")

        # todo fix the weighted version of loss and parameter passing
        model.compile(optimizer=training_utils.get_optimizer(test.optim, lr=float(test.lr), decay=float(test.decay)),
                      loss=getattr(losses, test.loss),
                      metrics=["accuracy"])

        train_dataset = test.train_dataset
        generator = get_train_generator(train_generator=test.train_generator,
                                        batch_size=test.batch,
                                        train_path=train_dataset,
                                        num_img=test.num_img,
                                        target_size=(test.input_size, test.input_size),
                                        aug=test.aug,
                                        num_classes=int(test.num_classes),
                                        classes=test.classes.split(","),
                                        folds=test.folds.split(","),
                                        cell_types=str(test.cell_types).split(","),
                                        )

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
            callbacks=[train_check, TB_callback, loss_recorder],  # checkpoint, plot_callback], #, earlyStopping],
            epochs=test.epoch,
            verbose=1)

        mlflow.log_artifacts(f"./tests_runs/{sheet}/{test.name}/")
        mlflow.keras.save_model(model, f"./tests_runs/{sheet}/{test.name}/saved_model", custom_objects=all_losses)

        ##SAVE MODEL###################
        model.save(f"./tests_runs/{sheet}/{test.name}/{sheet}_{test.name}_model.h5")

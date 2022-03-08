import os

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print(f"{path} already exists")

def create_experiment_folder(test_name, sheet):
    create_folder("./tests_runs/{}".format(sheet))
    create_folder("./tests_runs/{}/{}".format(sheet, test_name))
    create_folder("./tests_runs/{}/{}/training_images".format(sheet, test_name))
    create_folder("./tests_runs/{}/{}/test_results".format(sheet, test_name))
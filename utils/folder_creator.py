import os

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno is not os.errno.EEXIST:
            raise
        print("ERROR when creating {}".format(path))
        pass
    else:
        print("OK {} ".format(path))

def create_test_folder(test_name, sheet):
    create_folder("./tests/{}".format(sheet))
    create_folder("./tests/{}/{}".format(sheet, test_name))
    create_folder("./tests/{}/{}/training_images".format(sheet, test_name))
    create_folder("./tests/{}/{}/test_results".format(sheet, test_name))
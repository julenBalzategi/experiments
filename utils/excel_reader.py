import pandas as pd

class ExcelReader:

    def __init__(self, test_path, sheet):
        self.data = pd.ExcelFile(test_path).parse(sheet)
        self.items = self.data.iloc[:,0]
        self.data = self.data.loc[:, (self.data == "Y").any()]
        self.tests = self.data.columns.values
        self.idx_test = 0

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.tests) == self.idx_test:
            raise StopIteration
        else:
            self.idx_test += 1
            class Test(object):
                pass
            retTest = Test()
            data = self.data[self.tests[self.idx_test - 1]]
            for key, value in zip(self.items, data):
                retTest.__setattr__(key, value)
            return retTest

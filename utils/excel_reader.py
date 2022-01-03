import pandas as pd

class ExcelReader:

    def __init__(self, test_path, sheet):
        self.data = pd.ExcelFile(test_path).parse(sheet)
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
            return self.data[self.tests[self.idx_test - 1]]


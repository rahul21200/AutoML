from pycaret.classification import *


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sb


class DataPreProcessing:
    def __init__(self, df=None, path=None):
        # if path != None:
        #     self.df = pd.read_csv(path)
        # elif df != None:
        self.df = df
        # print(self.df)
        self.conditions = {
            "remove_outliers": False,
            # "fix_imbalance": False,
            "normalize": False,
            "feature_selection": False,
            "remove_multicollinearity": False,
            "pca": False
        }

        self.methods = {
            'outliers_threshold': 0.05,
            'normalize_method': 'zscore',
            'feature_selection_threshold': 0.8,
            'multicollinearity_threshold': 0.9,
            'pca_components': 0.99,
        }
        self.target = None

    def setTarget(self, target):
        self.target = target

    def outliers_threshold(self, val):
        """
        Default value is 0.05
        Ranges from 0 to 1
        """
        self.methods['outliers_threshold'] = val

    def normalize_method(self, meth):
        """
        Default: zscore
        Methods:
            1.zscore
            2.minmax
            3.maxabs
            3.robust
        """
        self.methods['normalize_method'] = meth

    def feature_selection_threshold(self, val):
        """
        Default value = 0.8
        """
        self.methods['feature_selection_threshold'] = val

    def multicollinearity_threshold(self, val):
        """
        Default value = 0.9
        """
        self.methods['multicollinearity_threshold'] = val

    def pca_components(self, val):
        """
        Default value = 0.99
        """
        self.methods['pca_components'] = val

    def data_stat(self):
        # print(self.df.info())
        # print(self.df.describe())
        pass

    def remove_outliers(self):
        self.conditions['remove_outliers'] = True

    def fix_imbalance(self):
        self.conditions['fix_imbalance'] = True

    def normalize(self):
        self.conditions['normalize'] = True

    def feature_selection(self):
        self.conditions['feature_selection'] = True

    def remove_multicollinearity(self):
        self.conditions['remove_multicollinearity'] = True

    def pca(self):
        self.conditions['pca'] = True

    def allSetup(self):
        model = setup(data=self.df, target=self.target,
                      **self.conditions, **self.methods, silent=True,verbose=False)
        return model

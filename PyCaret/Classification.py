import pycaret as pc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sb
from pycaret.classification import *


class classification:
    def __init__(self, path):
        self.df = pd.read_csv(path, header=0)
        self.target = None

    def initiate_target(self, target):
        self.target = target
        return target

    def MakeModel(self):
        setup(data=self.df, target=self.target)
        models = compare_models(exclude=[self.target])
        # print(models)
        return models

    def save(self, model_method):
        model = create_model(model_method)
        save_model(model)

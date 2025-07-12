# Note not every import is used here

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Apply the default theme
sns.set_theme(rc={"patch.force_edgecolor": False})

import os
import wget
from pathlib import Path
import shutil
import gzip

import re

pd.set_option('display.max_columns', None)

import random


from IPython.display import display


data = pd.read_csv("../HIGGS/HIGGS.csv", header=None)

COLUMNS_LIST = ["target", "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
data.columns = COLUMNS_LIST

data["target"] = data["target"].astype(int)

display(data)

OUTPUT_PATH = "../HIGGS/processed.pkl"

data.to_pickle(OUTPUT_PATH)




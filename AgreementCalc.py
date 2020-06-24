import tkinter as tk
from tkinter import filedialog
import fnmatch
import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import csv
import re
import glob
import pandas as pd

# Ranks
names = []
consistent = []
unique = []
ranks_ndp = []
root = tk.Tk()
root.withdraw()
filepath_csv = tk.filedialog.askdirectory(title='Choose where the csv file is')
os.chdir(filepath_csv)

files_csv = glob.glob("*.csv")

rankers = ['LBE', 'ADK', 'eh', 'JJ']

for i in range(len(rankers)):
    files = glob.glob('Results_'+rankers[i]+'*.csv')

    ranks = []
    for file in files:
        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                ranks.append(row)

    ranks = np.array(ranks, dtype=np.int)
    ranks = ranks[ranks[:, 2].argsort()]

    ranks = pd.DataFrame.drop_duplicates(pd.DataFrame(ranks, dtype=int, columns=['Better', 'Worse', 'ID']))

    exec('ranks_%s = ranks' % (rankers[i]))

ranks_all = pd.concat([ranks_LBE, ranks_ADK, ranks_eh, ranks_JJ], ignore_index=True)
ranks_all = ranks_all.sort_values(by=['ID'])

# Only keeps duplicated IDs
ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

ranks_same = ranks_dup[ranks_dup.duplicated(subset=['Better', 'Worse','ID'], keep=False)]

# For each ranker, removed self-agreement while keeping self-contradiction -> combine to find duplicates
print(f'Number of repeated pairs = {len(ranks_dup)}, same choices = {len(ranks_same)}, Agreement = {len(ranks_same)/len(ranks_dup)}')



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

with open(r"I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\file.txt", "r") as log_file:
    log = log_file.readlines()

contrast_all = [log[i].split('_')[6] for i in range(2920)]

# Ranks
names = []
consistent = []
unique = []
ranks_ndp = []

filepath_csv = r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020'
os.chdir(filepath_csv)
files_csv = glob.glob("*.csv")

rankers = ['JS']
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

    ranks = pd.DataFrame(ranks, dtype=int, columns=['Better', 'Worse', 'ID'])

    # Merge results for easier probability calc
    ranks['Results'] = ranks['Better']*10 + ranks['Worse']
    ranks = ranks.drop(['Better', 'Worse'], axis=1)

    # Probability of a specific result for each reviewer
    # This also removes same results when the image pair was shown multiple times.
    ranks = ranks.groupby('ID')
    ranks = ranks.apply(lambda x: x.groupby('Results').count() / x.shape[0]).rename(columns={'ID': 'Prob'}).reset_index()

    contrast = [contrast_all[i-1] for i in (ranks['ID'].to_numpy())]
    ranks['Contrast'] = contrast

    # merge T1 and T1PRE
    ranks.loc[ranks['Contrast'] =='AXT1PRE', 'Contrast'] = 'AXT1'

    ranks['Reviewer'] = rankers[i]

    exec('ranks_%s = ranks' % (rankers[i]))
    exec('ranks_%s_all = ranks' % (rankers[i]))

    # Likert
    files_likert = glob.glob('Likert_' + rankers[i] + '*.csv')[0]

    likert = []
    with open(files_likert) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            likert.append(row)

    likert = np.array(likert, dtype=np.int)
    likert = likert[np.lexsort((likert[:, 0], likert[:, 1]))]

    # Convert score to result
    results = []
    num_pairs_likert = likert.shape[0]//2
    for j in range(num_pairs_likert):
        score_delta = likert[j+num_pairs_likert, 2] - likert[j, 2]
        if score_delta < 0:
            results.append(0)
        elif score_delta ==0:
            results.append(22)
        else:
            results.append(10)

    likert = pd.DataFrame(likert, dtype=int, columns=['ID', 'PairID', 'Score', 'Time'])
    likert['Results'] = results+results

    contrast_likert = [contrast_all[k-1] for k in (likert['ID'].to_numpy())]
    likert['Contrast'] = contrast_likert

    # merge T1 and T1PRE
    likert.loc[likert['Contrast'] =='AXT1PRE', 'Contrast'] = 'AXT1'

    likert['Prob'] = 1
    likert['Reviewer'] = rankers[i]

    exec('likert_%s = likert' % (rankers[i]))
    exec('likert_%s_all = likert' % (rankers[i]))

# contrast_types = ['AXT1', 'AXT2', 'AXT1POST', 'AXFLAIR']
# ALL_C = True
#
# if ALL_C:
#
#
# else:

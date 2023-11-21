"""
This file is for calculating inter- intra- and overall agreement for each contrast types.
"""

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
from utils.utils import pad_beginning, add_zero_prob

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

# This is the order of radiologists A-G
rankers = ['LBE', 'ADK', 'JJ', 'eh', 'AP','THO','JS']

# I merged T1 and T1PRE
contrast_types = ['AXT1', 'AXT2', 'AXT1POST', 'AXFLAIR']
# contrast_types = ['AXT1POST']

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

ALL_C = False
INTER2 = False
if INTER2:
    if ALL_C:
        SELF = True

        if SELF:
            reviewer1 = ranks_LBE
            reviewer2 = reviewer1

        else:
            # Need to manually set to which reviewers
            ranks1 = ranks_JJ
            ranks2 = ranks_ADK
            ranks_all = pd.concat([ranks1, ranks2], ignore_index=True)
            ranks_all = ranks_all.sort_values(by=['ID'])

            # dump images only appeared once in ranks_all
            ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

            # split and get intersection of reviewed image IDs
            reviewer1 = ranks_dup[ranks_dup['Reviewer'] == 'JJ'].sort_values(['ID', 'Results'])
            reviewer2 = ranks_dup[ranks_dup['Reviewer'] == 'ADK'].sort_values(['ID', 'Results'])

            # keep only images reviewed by both
            id_1 = reviewer1.ID.isin(reviewer2.ID)
            reviewer1 = reviewer1[id_1]
            id_2 = reviewer2.ID.isin(reviewer1.ID)
            reviewer2 = reviewer2[id_2]

        num_intersection = len(reviewer1.drop_duplicates(subset=['ID']))
        print(f'Cross all contrasts: {num_intersection} pairs were ranked by both reviewers')

        # Pad zero probability to align
        reviewer1 = add_zero_prob(reviewer1, num_intersection)
        reviewer2 = add_zero_prob(reviewer2, num_intersection)

        # Agreement calc
        reviewer1_prob = reviewer1['Prob'].to_numpy()
        reviewer2_prob = reviewer2['Prob'].to_numpy()

        # Pad beginning
        # This may cause extra 0 (from extra nan) at the end, but it doesn't matter
        reviewer1_prob = pad_beginning(reviewer1, reviewer1_prob)
        reviewer2_prob = pad_beginning(reviewer2, reviewer2_prob)
        num_agreed = np.dot(reviewer1_prob[:num_intersection * 3], reviewer2_prob[:num_intersection * 3])

        # Agreed%
        percent_agreed = num_agreed / num_intersection

        print(f'Cross all contrasts: {percent_agreed * 100}% agreement between reviewers')

    else:
        for cc in contrast_types:

            SELF = False

            if SELF:
                reviewer1 = ranks_JS.loc[ranks_JS['Contrast'] == cc]
                reviewer2 = reviewer1.copy()

            else:
                # Need to manually set to which reviewers
                ranks1 = ranks_THO.loc[ranks_THO['Contrast'] == cc]
                ranks2 = ranks_JS.loc[ranks_JS['Contrast'] == cc]
                ranks_all = pd.concat([ranks1, ranks2], ignore_index=True)
                ranks_all = ranks_all.sort_values(by=['ID'])

                # dump images only appeared once in ranks_all
                ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

                # split and get intersection of reviewed image IDs
                reviewer1 = ranks_dup[ranks_dup['Reviewer'] == 'THO'].sort_values(['ID', 'Results'])
                reviewer2 = ranks_dup[ranks_dup['Reviewer'] == 'JS'].sort_values(['ID', 'Results'])

                # keep only images reviewed by both
                id_1 = reviewer1.ID.isin(reviewer2.ID)
                reviewer1 = reviewer1[id_1]
                id_2 = reviewer2.ID.isin(reviewer1.ID)
                reviewer2 = reviewer2[id_2]

            num_intersection = len(reviewer1.drop_duplicates(subset=['ID']))
            print(f'Contrast_{cc}: {num_intersection} pairs were ranked by both reviewers')

            # Pad zero probability to align
            reviewer1 = add_zero_prob(reviewer1, num_intersection)
            reviewer2 = add_zero_prob(reviewer2, num_intersection)

            # Agreement calc
            reviewer1_prob = reviewer1['Prob'].to_numpy()
            reviewer2_prob = reviewer2['Prob'].to_numpy()

            # Pad beginning
            # This may cause extra 0 (from extra nan) at the end, but it doesn't matter
            reviewer1_prob = pad_beginning(reviewer1, reviewer1_prob)
            reviewer2_prob = pad_beginning(reviewer2, reviewer2_prob)
            num_agreed = np.dot(reviewer1_prob[:num_intersection * 3], reviewer2_prob[:num_intersection * 3])

            # Agreed%
            percent_agreed = num_agreed / num_intersection

            print(f'Contrast_{cc}: {percent_agreed * 100}% agreement between reviewers')

else:
    if ALL_C:
        ranks_all = pd.concat([ranks_LBE, ranks_ADK, ranks_JJ, ranks_eh, ranks_AP, ranks_THO, ranks_JS], ignore_index=True)
        ranks_all = ranks_all.sort_values(by=['ID'])

        ranks_unique = ranks_all.drop_duplicates(subset=['ID'])

        print(f'Contrast {cc}: total number of image pairs {len(ranks_unique)}')

        # Only keeps duplicated IDs
        ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

        ranks_same = ranks_dup[ranks_dup.duplicated(subset=['Results', 'ID'], keep=False)]

        # # NOTE: for consensus.csv, we want to include those only appeared once, but for agreement calc, we don't want it for neither majority nor all votes.
        # # "clean labels", pairs with 2 or more votes + pairs that only appeared once
        # ranks_once = ranks_all.merge(ranks_dup, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        # ranks_once = ranks_once.drop(['_merge'], axis=1)
        #
        # ranks_clean = pd.concat([ranks_once, ranks_same], ignore_index=True)
        ranks_clean = ranks_same.copy()
        ranks_clean = ranks_clean.sort_values(by=['ID'])

        ranks_clean = ranks_clean.drop(['Prob', 'Reviewer'], axis=1)
        ranks_clean['Better'] = ranks_clean['Results']
        ranks_clean['Worse'] = ranks_clean['Results']


        ############################################### Calc overall agreement #############################################
        test = ranks_clean.groupby('ID')
        test = test['Better'].value_counts()

        idx_dup = []
        temp = test.index
        for i in range(len(test)):
            idx_dup.append(temp[i][0])
        # to remove duplicated from list
        idx = []
        [idx.append(x) for x in idx_dup if x not in idx]

        majority = 0
        for i in idx:
            if len(test[i])==1:
                majority += test.loc[i].values.item()
            elif len(test[i])==2:
                index0 = test[i].index[0]
                index1 = test[i].index[1]
                if test[i, index0] != test[i, index1]:
                    majority += max(test[i, index0], test[i, index1])
            elif len(test[i]) == 3:
                index0 = test[i].index[0]
                index1 = test[i].index[1]
                index2 = test[i].index[2]
                if test[i, index0] == test[i, index1] and test[i, index0] == test[i, index2]:
                    majority += 0
                else:
                    majority += max(test[i, index0], test[i, index1], test[i, index2])

        print(f'Overall agreement {majority / len(ranks_clean)}')
    else:
        for cc in contrast_types:
            for i in range(len(rankers)):
                exec('ranks_all = ranks_%s_all' % (rankers[i]))
                ranks = ranks_all.loc[ranks_all['Contrast'] == cc]
                exec('ranks_%s = ranks' % (rankers[i]))

            ranks_all = pd.concat([ranks_LBE, ranks_ADK, ranks_JJ, ranks_eh, ranks_AP, ranks_THO, ranks_JS],
                                  ignore_index=True)
            ranks_all = ranks_all.sort_values(by=['ID'])


            # Only keeps duplicated IDs
            ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

            ranks_same = ranks_dup[ranks_dup.duplicated(subset=['Results', 'ID'], keep=False)]

            # # NOTE: for consensus.csv, we want to include those only appeared once, but for agreement calc, we don't want it for neither majority nor all votes.
            # # "clean labels", pairs with 2 or more votes + pairs that only appeared once
            # ranks_once = ranks_all.merge(ranks_dup, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
            # ranks_once = ranks_once.drop(['_merge'], axis=1)
            #
            # ranks_clean = pd.concat([ranks_once, ranks_same], ignore_index=True)
            ranks_clean = ranks_same.copy()
            ranks_clean = ranks_clean.sort_values(by=['ID'])

            ranks_clean = ranks_clean.drop(['Prob', 'Reviewer'], axis=1)
            ranks_clean['Better'] = ranks_clean['Results']
            ranks_clean['Worse'] = ranks_clean['Results']

            ############################################### Calc overall agreement #############################################
            test = ranks_clean.groupby('ID')
            test = test['Better'].value_counts()

            idx_dup = []
            temp = test.index
            for i in range(len(test)):
                idx_dup.append(temp[i][0])
            # to remove duplicated from list
            idx = []
            [idx.append(x) for x in idx_dup if x not in idx]

            majority = 0
            for i in idx:
                if len(test[i]) == 1:
                    majority += test.loc[i].values.item()
                elif len(test[i]) == 2:
                    index0 = test[i].index[0]
                    index1 = test[i].index[1]
                    if test[i, index0] != test[i, index1]:
                        majority += max(test[i, index0], test[i, index1])
                elif len(test[i]) == 3:
                    index0 = test[i].index[0]
                    index1 = test[i].index[1]
                    index2 = test[i].index[2]
                    if test[i, index0] == test[i, index1] and test[i, index0] == test[i, index2]:
                        majority += 0
                    else:
                        majority += max(test[i, index0], test[i, index1], test[i, index2])

            print(f'Contrast {cc}: Overall agreement {majority / len(ranks_clean)}')
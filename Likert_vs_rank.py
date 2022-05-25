import tkinter as tk
from tkinter import filedialog
import fnmatch
import os
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import csv
import re
import glob
import pandas as pd
from utils.utils import pad_beginning, add_zero_prob


def compare_likert(likert2, i, score, agreement):
    if likert2.iloc[i, 2] == 1:
        agreement[0, score-1] += 1
    elif likert2.iloc[i, 2] == 2:
        agreement[1, score-1] += 1
    elif likert2.iloc[i, 2] == 3:
        agreement[2, score-1] += 1
    elif likert2.iloc[i, 2] == 4:
        agreement[3, score-1] += 1
    elif likert2.iloc[i, 2] == 5:
        agreement[4, score-1] += 1


# Quadratic weights for cohen's kappa
weights = np.array([[1, 0.94, 0.75, 0.44, 0],
                    [0.94, 1, 0.94, 0.75, 0.44],
                    [0.75, 0.94, 1, 0.94, 0.75],
                    [0.44, 0.75, 0.94, 1, 0.94],
                    [0, 0.44, 0.75, 0.94, 1]])

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

rankers = ['JS', 'AP']
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
            results.append(1)
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
    exec('likert_%s_all = likert' % (rankers[i]))
    exec('likert_%s = likert' % (rankers[i]))



contrast_types = ['AXT1', 'AXT2', 'AXT1POST', 'AXFLAIR']
ALL_C = False
LIKERT_V_RANK = True
if LIKERT_V_RANK:
    # compare a single reader's agreement on Likert and ranking
    for rr in rankers:
        exec('likert = likert_%s' % (rr))
        exec('ranks = ranks_%s' % (rr))

        if ALL_C:
            ranks_fromLikert = likert[['ID', 'Results', 'Prob', 'Reviewer']].copy()
            ranks_fromLikert = ranks_fromLikert[:num_pairs_likert]

            id = ranks.ID.isin(ranks_fromLikert.ID)
            ranks = ranks[id]

            num_intersection = len(ranks_fromLikert)
            ranks = add_zero_prob(ranks, num_intersection=num_intersection)
            ranks_fromLikert = add_zero_prob(ranks_fromLikert, num_intersection=num_intersection)

            ranks_prob = ranks['Prob'].to_numpy()
            ranks_fromLikert_prob = ranks_fromLikert['Prob'].to_numpy()

            ranks_prob = pad_beginning(ranks, ranks_prob)
            ranks_fromLikert_prob = pad_beginning(ranks_fromLikert, ranks_fromLikert_prob)
            ranks_prob = ranks_prob[:num_intersection*3]
            ranks_fromLikert_prob = ranks_fromLikert_prob[:num_intersection*3]

            tot_agreed = np.dot(ranks_prob, ranks_fromLikert_prob)
            agreed22 = np.dot(ranks_prob[2::3], ranks_fromLikert_prob[2::3])

            ranks_prob110 = np.delete(ranks_prob, np.s_[2::3])
            ranks_fromLikert_prob110 = np.delete(ranks_fromLikert_prob, np.s_[2::3])
            agreed110 = np.dot(ranks_prob110, ranks_fromLikert_prob110)

            print(f'Reader {rr}---------------------------------------------------------------------------------------')
            print(f'Total number of agreed pair {tot_agreed}')
            print(f'Agreement on similar pair {agreed22}')
            print(f'Agreement on one is better than the other {agreed110}')
            print('------------------------------------------------------------------------------------------------')

        else:
            for cc in contrast_types:
                ranks_c = ranks.loc[ranks['Contrast'] == cc]

                ranks_fromLikert = likert[['ID', 'Results', 'Prob', 'Reviewer', 'Contrast']].copy()
                ranks_fromLikert = ranks_fromLikert[:num_pairs_likert]
                ranks_fromLikert = ranks_fromLikert.loc[ranks_fromLikert['Contrast'] == cc]

                id = ranks_c.ID.isin(ranks_fromLikert.ID)
                ranks_c = ranks_c[id]

                num_intersection = len(ranks_fromLikert)
                ranks_c = add_zero_prob(ranks_c, num_intersection=num_intersection)
                ranks_fromLikert = add_zero_prob(ranks_fromLikert, num_intersection=num_intersection)

                ranks_prob = ranks_c['Prob'].to_numpy()
                ranks_fromLikert_prob = ranks_fromLikert['Prob'].to_numpy()

                ranks_prob = pad_beginning(ranks_c, ranks_prob)
                ranks_fromLikert_prob = pad_beginning(ranks_fromLikert, ranks_fromLikert_prob)
                ranks_prob = ranks_prob[:num_intersection * 3]
                ranks_fromLikert_prob = ranks_fromLikert_prob[:num_intersection * 3]

                tot_agreed = np.dot(ranks_prob, ranks_fromLikert_prob)
                agreed22 = np.dot(ranks_prob[2::3], ranks_fromLikert_prob[2::3])

                ranks_prob110 = np.delete(ranks_prob, np.s_[2::3])
                ranks_fromLikert_prob110 = np.delete(ranks_fromLikert_prob, np.s_[2::3])
                agreed110 = np.dot(ranks_prob110, ranks_fromLikert_prob110)

                print(f'Reader {rr}-----------------------------------------------------------------------------------')
                print(f'Contrast_{cc}: {num_intersection} pairs were both ranked and scored by reviewers')
                print(f'Total number of agreed pair {tot_agreed}')
                print(f'Agreement on similar pair {agreed22}')
                print(f'Agreement on one is better than the other {agreed110}')
                print('-----------------------------------------------------------------------------------------------')

else:
    # Compare the likert score from two readers.
    if ALL_C:

        for i in range(len(rankers)):
            exec('likert_temp = likert_%s_all' % (rankers[i]))
            num_images = likert_temp.groupby('Score').size()
            exec('num_images_%s = num_images' % (rankers[i]))
            # NOTE: Remember to check the scores they gave. e.g. JS didn't give any 1.

        agreement = np.zeros((5,5))
        for i in range(len(likert_JS_all)):
            if likert_JS_all.iloc[i, 2] == 1:
                compare_likert(likert_AP_all, i, 1, agreement)
            elif likert_JS_all.iloc[i, 2] == 2:
                compare_likert(likert_AP_all, i, 2, agreement)
            elif likert_JS_all.iloc[i, 2] == 3:
                compare_likert(likert_AP_all, i, 3, agreement)
            elif likert_JS_all.iloc[i, 2] == 4:
                compare_likert(likert_AP_all, i, 4, agreement)
            elif likert_JS_all.iloc[i, 2] == 5:
                compare_likert(likert_AP_all, i, 5, agreement)

        print(agreement)

        po = np.sum(agreement/(num_pairs_likert*2) * weights)
        bychance = np.pad(num_images_JS.to_numpy(), (1,0), 'constant', constant_values=0) * \
                   np.expand_dims(num_images_AP.to_numpy(),1) / ((num_pairs_likert*2)**2)
        pe = np.sum(bychance * weights)
        kappa = (po-pe) / (1-pe)
        sd = np.sqrt((po*(1-po))/((1-pe)**2))
        z = 1.96 # for 95% confidence level
        print(f'Quadratic weighted kappa={kappa}, sd {sd}, plus-minus {z*sd} for CI 95%')
    else:
        for cc in contrast_types:
            for i in range(len(rankers)):
                exec('likert_temp = likert_%s_all' % (rankers[i]))
                likert_c = likert_temp.loc[likert_temp['Contrast'] == cc]
                num_images = likert_c.groupby('Score').size()
                exec('likert_%s_c = likert_c' % (rankers[i]))
                exec('num_images_%s = num_images' % (rankers[i]))

                # NOTE: Remember to check the scores they gave.
                # Pad zeros accordingly for bychance matrix calc
                print(f'{rankers[i]}, {num_images}')

            agreement = np.zeros((5, 5))
            for i in range(len(likert_JS_c)):
                if likert_JS_c.iloc[i, 2] == 1:
                    compare_likert(likert_AP_c, i, 1, agreement)
                elif likert_JS_c.iloc[i, 2] == 2:
                    compare_likert(likert_AP_c, i, 2, agreement)
                elif likert_JS_c.iloc[i, 2] == 3:
                    compare_likert(likert_AP_c, i, 3, agreement)
                elif likert_JS_c.iloc[i, 2] == 4:
                    compare_likert(likert_AP_c, i, 4, agreement)
                elif likert_JS_c.iloc[i, 2] == 5:
                    compare_likert(likert_AP_c, i, 5, agreement)
            print(agreement)
            agreement_prob = agreement / len(likert_AP_c)

            po = np.sum(agreement_prob * weights)
            num_images_JS_pad = np.pad(num_images_JS.to_numpy(), (2, 0), 'constant', constant_values=0)
            num_images_AP_pad = np.pad(num_images_AP.to_numpy(), (0, 0), 'constant', constant_values=0)
            bychance = num_images_JS_pad * np.expand_dims(num_images_AP_pad, 1) / (len(likert_AP_c) ** 2)
            pe = np.sum(bychance * weights)
            kappa = (po - pe) / (1 - pe)
            sd = np.sqrt((po * (1 - po)) / ((1 - pe) ** 2))
            z = 1.96  # for 95% confidence level
            print(f'{cc}: Quadratic weighted kappa={kappa}, sd {sd}, plus-minus {z * sd} for CI 95%')

# # plot time
# time_JS = likert['Time'].to_numpy()
# time_AP = likert['Time'].to_numpy()





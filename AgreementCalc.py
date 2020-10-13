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


def pad_beginning(data, prob):
    # data: dataframe
    # prob: 1d numpy array
    # fix bug when dataframe doen't start with [ID,1,Prob, Reviewer]
    if data.iloc[0]['Results'] == 10:
        prob = np.insert(prob,0, 0, axis=0)
    elif data.iloc[0]['Results'] == 22:
        prob = np.insert(prob,0, (0,0), axis=0)
    else:
        pass
    return prob


def add_zero_prob(reviewerdata):

    reviewer_results = reviewerdata['Results'].to_numpy()
    reviewer_ID = reviewerdata['ID'].to_numpy()

    idx_incomp = []
    jj = np.arange(0, len(reviewer_results), 1)
    j = 0
    idx_incomp.append(j)
    for i in range(1, len(reviewer_results)):
        if reviewer_ID[i] == reviewer_ID[i - 1]:
            if reviewer_results[i - 1] == 1:
                if reviewer_results[i] == 10:
                    j += 1
                else:
                    j += 2
            else:
                j += 1
        else:
            if reviewer_results[i - 1] == 1:
                if reviewer_results[i] == 1:
                    j += 3
                elif reviewer_results[i] == 10:
                    j += 4
                else:
                    j += 5
            elif reviewer_results[i - 1] == 10:
                if reviewer_results[i] == 1:
                    j += 2
                elif reviewer_results[i] == 10:
                    j += 3
                else:
                    j += 4
            else:
                if reviewer_results[i] == 1:
                    j += 1
                elif reviewer_results[i] == 10:
                    j += 2
                else:
                    j += 3

        idx_incomp.append(j)
    idx_incomp = np.array(idx_incomp)

    reviewerdata['j'] = idx_incomp
    reviewerdata = reviewerdata.set_index('j')

    # this will fill missing data as nan
    reviewerdata = reviewerdata.reindex(index=np.arange(num_intersection * 3))
    reviewerdata['Prob'].fillna(0, inplace=True)

    return reviewerdata


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

    ranks = pd.DataFrame(ranks, dtype=int, columns=['Better', 'Worse', 'ID'])

    # Merge results for easier probability calc
    ranks['Results'] = ranks['Better']*10 + ranks['Worse']
    ranks = ranks.drop(['Better', 'Worse'], axis=1)

    # Probability of a specific result for each reviwer
    # This also removes same results when the image pair was shown multiple times.
    ranks = ranks.groupby('ID')
    ranks = ranks.apply(lambda x: x.groupby('Results').count() / x.shape[0]).rename(columns={'ID': 'Prob'}).reset_index()

    ranks['Reviewer'] = rankers[i]

    exec('ranks_%s = ranks' % (rankers[i]))

SELF = True

if SELF:
    reviewer1 = ranks_eh
    reviewer2 = reviewer1

else:

    # Need to manually set to which reviewers
    ranks_all = pd.concat([ranks_LBE, ranks_ADK], ignore_index=True)
    ranks_all = ranks_all.sort_values(by=['ID'])

    # Remove those reviewed only once by either reviewers
    ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]

    # split and get intersection of reviewed image IDs
    reviewer1 = ranks_dup[ranks_dup['Reviewer'] == 'LBE'].sort_values(['ID', 'Results'])
    reviewer2 = ranks_dup[ranks_dup['Reviewer'] == 'ADK'].sort_values(['ID', 'Results'])

    id_1 = reviewer1.ID.isin(reviewer2.ID)
    reviewer1 = reviewer1[id_1]
    id_2 = reviewer2.ID.isin(reviewer1.ID)
    reviewer2 = reviewer2[id_2]


num_intersection = len(reviewer1.drop_duplicates(subset=['ID']))
print(f'{num_intersection} pairs were ranked by both reviewers')

# Pad zero probability to align
reviewer1 = add_zero_prob(reviewer1)
reviewer2 = add_zero_prob(reviewer2)


# Agreement calc
reviewer1_prob = reviewer1['Prob'].to_numpy()
reviewer2_prob = reviewer2['Prob'].to_numpy()

# Pad beginning
# This may cause extra 0 (from extra nan) at the end, but it doesn't matter
reviewer1_prob = pad_beginning(reviewer1, reviewer1_prob)
reviewer2_prob = pad_beginning(reviewer2, reviewer2_prob)
num_agreed = np.dot(reviewer1_prob[:num_intersection*3],reviewer2_prob[:num_intersection*3])

# Agreed%
percent_agreed = num_agreed/num_intersection

print(f'{percent_agreed*100}% agreement between reviewers')

# Bland-Altman
# values = np.tile([0,1, 0.5], num_intersection*3)
# values1 = []
# values2 = []
# for i in range(num_intersection):
#     temp1 = np.dot(values[i*3:i*3+3], reviewer1_prob[i*3:i*3+3])
#     temp2 = np.dot(values[i*3:i*3+3], reviewer2_prob[i*3:i*3+3])
#
#     values1.append(temp1)
#     values2.append(temp2)
#
# values1 = np.array(values1)
# values2 = np.array(values2)
#
# diff = values2 - values1
# mean = (values1 + values2)/2
#
# plt.scatter(mean, diff)
# plt.title('Bland-Altman Plot of ADK and ADK')
# plt.xlabel('Mean of two reviewers')
# plt.ylabel('Difference of two reviewers')
# plt.grid()
# plt.show()

#
# Clean up the label
# ranks_all = pd.concat([ranks_LBE, ranks_ADK, ranks_eh, ranks_JJ], ignore_index=True)
# ranks_all = ranks_all.sort_values(by=['ID'])
#
# # Only keeps duplicated IDs
# ranks_dup = ranks_all[ranks_all.duplicated(subset=['ID'], keep=False)]
#
# ranks_same = ranks_dup[ranks_dup.duplicated(subset=['Better', 'Worse','ID'], keep=False)]
#
# # "clean labels", pairs with 2 or more votes + pairs that only appeared once
# ranks_once = ranks_all.merge(ranks_dup, how='outer', indicator=True).loc[lambda x:x['_merge']=='left_only']
# ranks_once = ranks_once.drop(['_merge'], axis=1)
#
# ranks_clean = pd.concat([ranks_once, ranks_same], ignore_index=True)
# ranks_clean = ranks_clean.sort_values(by=['ID'])
#
# ranks_clean.to_csv('ranks_consensus.csv', index=False, header=False)

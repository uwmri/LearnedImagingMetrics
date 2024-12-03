import os
import glob
import csv
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import ttest_rel

####################### Ranking accuracies #######################
# TODO Add EfficientNet w/ and w/o subtraction
IQNet=[0.7564, 0.7537, 0.7646, 0.7581, 0.7276]
MSE=[0.7097, 0.6977, 0.7329, 0.7118, 0.6626]
SSIM=[0.6772, 0.6538, 0.6766, 0.696, 0.6292]

t_stat, p_value = ttest_rel(IQNet, MSE)
print(f"Paired T-Statistic: {t_stat}, P-Value: {p_value}")

# Perform the repeated measures ANOVA
f_stat, p_value = f_oneway(IQNet, MSE, SSIM)
print(f"F-Statistic: {f_stat}")
print(f"P-Value: {p_value}")

# Calculate mean of all values
all_data = np.array(IQNet + MSE + SSIM)
grand_mean = np.mean(all_data)

# Calculate SS Total
ss_total = np.sum((all_data - grand_mean) ** 2)

# Calculate SS Between
n = len(IQNet)  # Number of values in each group
ss_between = n * (np.sum((np.mean(IQNet) - grand_mean) ** 2) +
                  np.sum((np.mean(MSE) - grand_mean) ** 2) +
                  np.sum((np.mean(SSIM) - grand_mean) ** 2))

eta_squared = ss_between / ss_total
cohen_f = np.sqrt(eta_squared / (1 - eta_squared))

print(f"Cohen's f: {cohen_f}")

# ###### is the sample size big enough? ##########
# from statsmodels.stats.power import FTestAnovaPower
#
# # Parameters
# effect_size = cohen_f
# alpha = 0.05  # Significance level
# power = 0.8  # Desired power
# num_groups = 3  # Number of groups
#
# # Calculate sample size per group
# analysis = FTestAnovaPower()
# sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=num_groups)
# print(f"min sample_size: {sample_size}")
#########################################################################

####################### Ranking vs scoring time #######################
directory = r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020'
# JS
time_ranking = pd.read_csv(glob.glob(os.path.join(directory, 'Rank_vLikert_AP*.csv'))[0], names=['id1', 'id2', 'index', 'time'])
time_ranking = time_ranking.sort_values(by='index')

# Remove rows with time > 15s
outliers = time_ranking['index'][time_ranking['time'] > 15].tolist()
time_ranking = time_ranking[time_ranking['time'] <= 15]
time_ranking['time'] = time_ranking['time'] * 0.5
time_ranking = pd.concat([time_ranking] * 2, ignore_index=True)
time_ranking = time_ranking.sort_values(by='index')

time_likert = pd.read_csv(glob.glob(os.path.join(directory, 'Likert_AP*2022.csv'))[0], names=['index', 'id', 'score', 'time'])
time_likert = time_likert.sort_values(by='index')
time_likert = time_likert[~time_likert['index'].isin(outliers)]


# Perform paired t-test
t_stat, p_value = ttest_rel(time_likert['time'], time_ranking['time'])

# Print the results
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
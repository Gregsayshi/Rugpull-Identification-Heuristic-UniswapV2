import re
import sqlite3
import pandas as pd
import logging
import warnings
from pathlib import Path
import matplotlib.ticker as mtick
import matplotlib.dates as md
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
import datetime as dt
import numpy as np
warnings.filterwarnings('ignore')

path = r"(YOUR_PATH)\data\computer_generated\figures\rugpull_identification"


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



df = pd.read_csv("(YOUR_PATH)\data\computer_generated\rugpull_identification_results\rugpull_identification_results.csv",sep=",")

# Overall Metrics
df = df[df['is_rugpull'].notna()] #remove all non compliant trading pairs e.g. too few sync events & not WETH pairs
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#pair creation plot
df_pair_created = df[["pair_created_date"]].copy()
df_pair_created["pair_created"] = df_pair_created["pair_created_date"].astype(float)
df_pair_created["pair_created"] = df_pair_created["pair_created"].dropna()
dates=[dt.datetime.fromtimestamp(ts) for ts in df_pair_created["pair_created"].dropna()]
dates = df_pair_created["pair_created"].dropna()
datenums=md.epoch2num(dates)
values = df_pair_created.dropna().index
fig, ax = plt.subplots(figsize=(32,16))
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
plt.gca()
plt.hist(datenums,bins=369,color='#412d3a')
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.set_major_locator(md.MonthLocator())
ax.set_xlabel('Time')
ax.set_ylabel('Number Of Trading Pairs')
ax.set_title('Uniswap V2 Trading Pair Creation Over Time')
fig = ax.get_figure()
fig.savefig(str(path)+'\\Pair_Created_Over_Time_hist.pdf')
plt.close()
########################################################################################################
#Unique EOAs per pair plot
df_eoa_count = df[["eoas_count"]].copy()
df_eoa_count = df_eoa_count["eoas_count"].to_list()
df_eoa_count = np.asarray(df_eoa_count, dtype=np.float32)
filtered = df_eoa_count[~is_outlier(df_eoa_count)]
fig, ax = plt.subplots(figsize=(32,16))
plt.boxplot(filtered, vert=False, showfliers=False)
ax.set_xlabel('Count Of Unique EOAs That Interact With Trading Pair')
ax.set_title('Distribution Of Unique EOAs Per Trading Pair')
fig = ax.get_figure()
fig.savefig(str(path)+'\\EOAs_Count_Box.pdf')
plt.close()
########################################################################################################
#sync events per pair plot
df_sync_count = df[["sync_events_count"]].copy()
sync_events_count = df_sync_count["sync_events_count"].to_list()
sync_events_count = np.asarray(sync_events_count, dtype=np.float32)
filtered = sync_events_count[~is_outlier(sync_events_count)]
fig, ax = plt.subplots(figsize=(32,16))
plt.hist(filtered,bins=300,color='#412d3a')
ax.set_xlabel('Number Of Sync Events In Trading Pair')
ax.set_ylabel('Number Of Trading Pairs')
ax.set_title('Distribution Of Sync Events Counts Per Trading Pair')
fig = ax.get_figure()
fig.savefig(str(path)+'\\Sync_Events_Count_Hist.pdf')
plt.close()
########################################################################################################
#Fraction EOAs with negative balance per pair plot HISTOGRAM
df_losers = df[["frac_losers"]].copy()
x = df_losers.dropna().frac_losers
y = df_losers.dropna().index
fig, ax = plt.subplots(figsize=(32,16))
plt.hist(x,bins=100,color='#412d3a')
ax.set_xlabel('Fraction Of EOAs With A Negative Balance From Interactions With Trading Pair')
ax.set_ylabel('Number Of Trading Pairs')
ax.set_title('Distribution Of EOAs With A Negative WETH Balance Per Trading Pair')
fig = ax.get_figure()
fig.savefig(str(path)+'\\EOAs_Neg_Balance_Hist.pdf')
plt.close()
########################################################################################################
#Fraction EOAs with negative balance per pair plot BOXPLOT
df_losers = df[["frac_losers"]].copy()
x = df_losers.dropna().frac_losers
y = df_losers.dropna().index
fig, ax = plt.subplots(figsize=(32,16))
plt.boxplot(x, vert=False, showfliers=False)
ax.set_xlabel('Fraction Of EOAs With A Negative Balance From Interactions With Trading Pair')
ax.set_title('Distribution Of Losing EOAs Per Trading Pair')
fig = ax.get_figure()
fig.savefig(str(path)+'\\EOAs_Neg_Balance_Box.pdf')
plt.close()



# RUG PULL ESTIMATE
########################################################################################################
########################################################################################################
compliant = df.dropna(subset=["is_rugpull"])
df_lib = compliant # ONLY CONSIDER COMPLIANT TRADING PAIRS FOR FURTHER EVALUATION!!!
is_rug_lib = df_lib[df_lib["is_rugpull"]==1]

#pair creation / rug-pull time overlay plot
is_rug_rug_time_lib = is_rug_lib[["rug_time"]].copy()
is_rug_rug_time_lib["rug_time"] = is_rug_rug_time_lib["rug_time"].astype(float)
is_rug_rug_time_lib["rug_time"] = is_rug_rug_time_lib["rug_time"].dropna()
dates_lib=[dt.datetime.fromtimestamp(ts) for ts in is_rug_rug_time_lib["rug_time"].dropna()]
dates_lib = is_rug_rug_time_lib["rug_time"].dropna()
datenums_rug_lib=md.epoch2num(dates_lib)
datenums_rug = datenums_rug_lib

df_pair_created = df_lib[["pair_created_date"]].copy()
df_pair_created["pair_created"] = df_pair_created["pair_created_date"].astype(float)
df_pair_created["pair_created"] = df_pair_created["pair_created"].dropna()
dates=[dt.datetime.fromtimestamp(ts) for ts in df_pair_created["pair_created"].dropna()]
dates = df_pair_created["pair_created"].dropna()
datenums_create=md.epoch2num(dates)


fig, ax = plt.subplots(figsize=(32,16))
plt.subplots_adjust(bottom=0.2)
plt.xticks( rotation=25 )
plt.gca()
plt.hist(datenums_create,bins=369,color='#412d3a', label="Pair Deployment")
plt.hist(datenums_rug,bins=369,color='#df4232', label="Rug-Pull Estimate")
plt.legend(loc="upper left")
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.set_major_locator(md.MonthLocator())
ax.set_xlabel('Time')
ax.set_ylabel('Number Of Trading Pairs')
ax.set_title('Uniswap V2 Trading Pair Creations and Rug-Pulls Over Time')
fig = ax.get_figure()
fig.savefig(str(path)+'\\liberal_Rug_Pulls_Over_Time_hist.pdf')
plt.close()


print("ALL DONE.")

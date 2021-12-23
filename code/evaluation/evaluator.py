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


df = pd.read_csv("(YOUR_PATH)\data\computer_generated\rugpull_identification_results\rugpull_identification_results.csv",sep=",")

has_weth_no = df[df["is_weth_pair"]==0] #filter for non-WETH pairs
too_low_syncs = df[df["low_syncs"]!=1] #filter for pairs with under 10 sync events
low_syncs = len(too_low_syncs.index) / len(df.index)
is_resampled = df[df["is_rugpull"]==1]
no_weth = len(has_weth_no.index) / len(df.index)
print("\nFraction of Trading Pairs without WETH in one of their pools:"+str(no_weth))
print("\nFraction of Trading Pairs with less than 10 transactions:"+str(low_syncs))

compliant = df.dropna(subset=["is_rugpull"])
df = compliant # ONLY CONSIDER COMPLIANT TRADING PAIRS FOR FURTHER EVALUATION!!!
print("\nNumber compliant samples:"+str(len(df.index)))
print("\nFraction non compliant samples:"+str((len(df.index)/len(df.index))))


print("\n\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("ESTIMATE RESULTS EVALUATION")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
is_rug = df[df["is_rugpull"]==1]
print("\n\nNumber Trading Pairs that are subject to a rug-pull before EOA>1:"+str(len(is_rug.index)))
is_rug = is_rug[is_rug["eoas_count"] > 1] #remove pairs with only 1 EOA
print("\nNumber Trading Pairs that are subject to a rug-pull after EOA>1:"+str(len(is_rug.index)))
is_rug_lib = is_rug
frac_rug_pulls = len(is_rug.index) / len(df["is_rugpull"].index)
print("\nFraction of Trading Pairs that are subject to a rug-pull:"+str(frac_rug_pulls))

df_first_months = df[df["pair_created_date"] < 1596232800]
print("\n\n\nTrading Pairs created before 01/08/2020: "+str(len(df_first_months.index)))
is_rug_first_months = is_rug[is_rug["rug_time"] < 1596232800]
print("Trading Pairs that are subject to a rug-pull before 01/08/2020: "+str(len(is_rug_first_months.index)))
print("rug-to-creation ratio before 01/08/2020: "+str(len(is_rug_first_months.index)/len(df_first_months.index)))

df_middle = df[df["pair_created_date"] < 1604188800]
df_middle = df_middle[df_middle["pair_created_date"] > 1596232800]
print("\nTrading Pairs created after after 01/08/2020 and before 01/11/2020: "+str(len(df_middle.index)))
is_rug_middle = is_rug[is_rug["rug_time"] < 1604188800]
is_rug_middle = is_rug_middle[is_rug_middle["rug_time"] > 1596232800]
print("Trading Pairs that are subject to a rug-pull after 01/08/2020 and before 01/11/2020: "+str(len(is_rug_middle.index)))
print("rug-to-creation ratio after 01/08/2020 and before 01/11/2020: "+str(len(is_rug_middle.index)/len(df_middle.index)))

df_winter = df[df["pair_created_date"] > 1604188800]
print("\nTrading Pairs created after 01/11/2020: "+str(len(df_winter.index)))
is_rug_winter = is_rug[is_rug["rug_time"] > 1604188800]
print("Trading Pairs that are subject to a rug-pull after 01/11/2020: "+str(len(is_rug_winter.index)))
print("rug-to-creation ratio after 01/11/2020: "+str(len(is_rug_winter.index)/len(df_winter.index)))

print("\nSync Evets Description in rug-pull Pairs: "+str((is_rug["sync_events_count"].describe())))
is_rug_sorted = is_rug.sort_values(by="sync_events_count",ascending=False) #sort by number of sync Events
is_rug_sorted = is_rug_sorted.head(20)
print("\nHighest Sync event counts rug-pull addresses:")
for addr in is_rug_sorted["pair_address"]:
    print(addr)
print("\nHighest Sync event counts rug-pull sync counts:")
for val in is_rug_sorted["sync_events_count"]:
    print(val)
print("\nRug times:")
for val in is_rug_sorted["rug_time"]:
    print(val)
print("\nSize Compare:")
for val in is_rug_sorted["size_ratio"]:
    print(val)

print("\nHighest Sync event count in rug-pull\n: "+str((is_rug[is_rug["sync_events_count"] == is_rug["sync_events_count"].max()])))
print("\nHighest Sync event counts in rug-pull\n: "+str((is_rug.sort_values(by="sync_events_count"))))
df_rug_dur = is_rug[is_rug["dur_until_rug_hours"] > 0]
print("\nDuration until Rug Description: "+str((df_rug_dur["dur_until_rug_hours"].describe().apply(lambda x: format(x, 'f')))))
print("\nEOA count in rug-pull pairs Description\n: "+str((is_rug["eoas_count"].describe())))
print("\nHighest EOA count in rug-pull pairs Description\n: "+str((is_rug[is_rug["eoas_count"] == is_rug["eoas_count"].max()])))
print("\nLosing EOA frac in rug-pull pairs Description\n: "+str((is_rug["frac_losers"].describe())))
print("\nSize Ratio Description in rug-pull pairs: "+str((is_rug["size_ratio"].describe())))

print("ALL DONE")

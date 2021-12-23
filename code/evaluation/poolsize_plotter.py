import json
import re
import sqlite3
import pandas as pd
import logging
import warnings
from pathlib import Path
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import statistics
from statistics import StatisticsError
from datetime import datetime
from datetime import timedelta
import pickle as pkl
import multiprocessing as mp
import time
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt

def connect():
    """ Connects to DataBase """
    conn = sqlite3.connect(r'(YOUR_PATH)\data\base_dataset\uniswap_data.db')
    cur = conn.cursor()
    return conn, cur

########################################################################################################
########################################################################################################

def get_dlt_sync(df):
    """ Gets the deltas between sync events in order to make pool size changes tangable"""

    df["reserve0_current"] = df["reserve0"]
    df["reserve1_current"] = df["reserve1"]
    df["reserve0_previous"] = df["reserve0"].shift(1) #shift one up -> position(n) becomes position(n-1)
    df["reserve1_previous"] = df["reserve1"].shift(1)
    df["deltaABS_res0"] = df["reserve0_current"].astype(float) - df["reserve0_previous"].astype(float)
    df["deltaABS_res1"] = df["reserve1_current"].astype(float) - df["reserve1_previous"].astype(float)

    df = df.drop("reserve0_current",axis=1)
    df = df.drop("reserve1_current",axis=1)
    df = df.drop("reserve0_previous",axis=1)
    df = df.drop("reserve1_previous",axis=1)

    df.loc[df.index == 0, ["deltaABS_res0"]] = float(df["reserve0"][0]) #account for initial liquidity added during exchange pair creation -> add initial reserve as positive DeltaABS
    df.loc[df.index == 0, ["deltaABS_res1"]] = float(df["reserve1"][0])

    return df

########################################################################################################
########################################################################################################

def group_sync_events_by_pair(target):
    """ Queries all events for exchange contract, stores each event type in seperate dataframe.
        Returns dictionary of dataframes."""
    sync_df = None

    try:
        conn, cur = connect() # connect to the SQL server
        df_dict = {}

        try: #get pair table to see if pair is compliant to having ETH in one of the pools
            query = " SELECT * FROM pair WHERE pair_address = '%s' "
            records_to_insert = (target,)
            pair_df = pd.read_sql_query(query %records_to_insert, conn)
            pair_df = shorten_sample(pair_df)
            weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" #wrapped ETH (WETH) account address
            weth_in_1 = pair_df[pair_df["asset_1"] == weth]
            weth_in_2 = pair_df[pair_df["asset_2"] == weth]

            if weth_in_1.empty:
                if weth_in_2.empty:
                    sync_df = None #if no ETH in pools, disregard
                    return sync_df
                else:
                    weth_pos = 2
            else:
                weth_pos = 1

            #get sync events
            query = " SELECT * FROM sync WHERE address = '%s' "
            records_to_insert = (target,)
            sync_df = pd.read_sql_query(query %records_to_insert, conn)
            sync_df = get_dlt_sync(sync_df)
            sync_df["weth_pos"] = weth_pos
            creator_eoa = sync_df["from_address"][0] #first sync event = first transaction = initiated by contract creator eoa
            sync_df["creator"] = creator_eoa
        except Exception as e:
            #logger.error(f"Failed at group_events_by_pair() -> SYNC-events with the following error: {e}")
            print(f"Failed at group_sync_events_by_pair() with the following error:"+str(e))
    except Exception as e:
        #logger.error(f"Failed at group_events_by_pair() -> BURN-events with the following error: {e}")
        print(f"Failed at establishing DB connect @ group_sync_events_by_pair() with the following error: "+str(e))
    finally:
        conn.commit()
        conn.close()
        return sync_df

########################################################################################################
########################################################################################################

def shorten_sample(sync_df):
    """ Shortens target dataframe with sync events to save computational time needed.
        Until 2000 transactions -> analyse every transaction
        2000-2999 transactions -> analyse every 2. transaction
        3000-3999 transactions -> analyse every 3. transaction
        ..."""

    length = len(sync_df.index)
    for x in range(2000,10000000,1000):
        if length > x:
            continue
        if length <= x:
            if length < 2000: #for exchange pairs with <2000 transactions, use every transaction
                return sync_df
            else:
                if x < 10000:
                    shorter = int(str(x)[:1]) - 1
                    sync_df = sync_df.iloc[::shorter, :]
                    return sync_df
                if x < 100000:
                    shorter = int(str(x)[:2]) - 1
                    sync_df = sync_df.iloc[::shorter, :]
                    return sync_df
                if x < 1000000:
                    shorter = int(str(x)[:3]) - 1
                    sync_df = sync_df.iloc[::shorter, :]
                    return sync_df

########################################################################################################
############### >>> INSERT TARGET TRADING PAIR ADDRESS BELOW <<< #######################################
########################################################################################################
########################################################################################################
target_arr = ['insert_target_pair']
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


for ex_contr in target_arr:
        print("\n\nWorking on group_sync_events_by_pair() for contract: "+str(ex_contr))
        sync_df = group_sync_events_by_pair(ex_contr)
        short_sync_df = sync_df
        if sync_df is not None:
            #short_sync_df = shorten_sample(sync_df)
            if len(sync_df.index) > len(short_sync_df.index):
                print("Sync_df for contract "+str(ex_contr)+" shortend from "+str(len(sync_df.index))+" samples to "+str(len(short_sync_df.index))+" samples.")
            sync_df = short_sync_df



#def plot_poolchanges(df,realrugts,sus_dict):
def plot_poolchanges(df):
    ex_contr = df["address"][0]
    weth_pos = df["weth_pos"][0]

    pathSB = r"(YOUR_PATH)\data\computer_generated\figures\rugpull_identification"


    #get columns into right format
    def to_int(x):
        x = float(x)
        return x

    df["reserve0"] = df.apply(lambda x: to_int(x["reserve0"]), axis=1)
    df["reserve1"] = df.apply(lambda x: to_int(x["reserve1"]), axis=1)

    if weth_pos == 1:
        y = df.reserve0
    if weth_pos == 2:
        y = df.reserve1

    x = []
    for stamp in df["block_timestamp"]:
        time = datetime.utcfromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S')
        time = datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
        x.append(time)

    fig, ax = plt.subplots(figsize=(32,16))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.plot(x,y,color = '#412d3a')
    plt.gcf().autofmt_xdate()
    ax.set_xlabel('Time')
    ax.set_ylabel('WETH in pool')
    ax.set_title('Wrapped Ethereum (WETH) pool size for exchange contract: '+str(ex_contr))
    fig = ax.get_figure()

    fig = ax.get_figure()
    fig.savefig(str(pathSB)+'\\WETH_pool_size_BAR'+str(ex_contr)+'.pdf')
    plt.close()


plot_poolchanges(sync_df)

print("ALL DONE.")

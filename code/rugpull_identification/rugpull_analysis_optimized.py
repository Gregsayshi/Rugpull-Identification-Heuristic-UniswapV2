#Retrieve Sync Events and group them by exchange addresses
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
import cProfile, pstats

def init_analysis(pool_shrink_threshold,frac_losers_threshold,targets,db_path):#,out_put_file):

    def connect(db_path):
        """ Connects to DataBase """
        conn = sqlite3.connect(db_path) #rc)
        cur = conn.cursor()
        return conn, cur
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
                    if x < 10000000:
                        shorter = int(str(x)[:4]) - 1
                        sync_df = sync_df.iloc[::shorter, :]
                        return sync_df

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

    def group_sync_events_by_pair(target,db_path):
        """ Queries all events for exchange contract, stores each event type in seperate dataframe.
            Returns dictionary of dataframes."""
        sync_df = None
        short_sync_df = None

        conn, cur = connect(db_path) # connect to the SQL server
        df_dict = {}

        query = " SELECT * FROM pair WHERE pair_address = '%s' "
        records_to_insert = (target,)
        pair_df = pd.read_sql_query(query %records_to_insert, conn)

        weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2" #wrapped ETH (WETH) account address
        weth_in_1 = pair_df[pair_df["asset_1"] == weth]
        weth_in_2 = pair_df[pair_df["asset_2"] == weth]

        is_weth_pair = 1
        if weth_in_1.empty:
            if weth_in_2.empty:
                sync_df = None
                short_sync_df = None
                is_weth_pair = 0 #no WETH in either pool
            else:
                weth_pos = 2
        else:
            weth_pos = 1

        #get sync events
        query = " SELECT * FROM sync WHERE address = '%s' "
        records_to_insert = (target,)
        sync_df = pd.read_sql_query(query %records_to_insert, conn)
        read_sql_query_time = datetime.now() - startTime
        sync_events_count = len(sync_df.index)
        try:
            pair_created_date = sync_df["block_timestamp"][0]
        except IndexError:
            pair_created_date = None #If no sync events exist..
        is_resampled = 0
        sync_events_count_resampled = None
        resampling_decrease = None

        if is_weth_pair == 1: #preprocessing filter 1
            if sync_events_count >= 10: #preprocessing filter 2
                sync_df = get_dlt_sync(sync_df)
                sync_df["weth_pos"] = weth_pos
                creator_eoa = sync_df["from_address"][0] #first sync event = first transaction = initiated by contract creator eoa

                sync_df["creator"] = creator_eoa
                short_sync_df = shorten_sample(sync_df)

                if sync_events_count > len(short_sync_df.index):
                    print("Sync_df for contract "+str(ex_contr)+" shortend from "+str(len(sync_df.index))+" samples to "+str(len(short_sync_df.index))+" samples.")
                    is_resampled = 1
                    sync_events_count_resampled = len(short_sync_df.index)
                    resampling_decrease = 1 - ((sync_events_count_resampled) / (sync_events_count))


        conn.commit()
        conn.close()
        if is_weth_pair == 0:
            sync_df = None
            short_sync_df = None
        if sync_events_count < 10:
            sync_df = None
            short_sync_df = None
        return sync_df,short_sync_df,is_weth_pair,sync_events_count,pair_created_date,is_resampled,sync_events_count_resampled,resampling_decrease

    ########################################################################################################
    ########################################################################################################

    # Analyse transactions
    def find_rug_tr(pairdf,pool_shrink_threshold):
        result_dict = {}
        if pairdf["weth_pos"][0] == 1:
            weth_pos = 1
        else:
            weth_pos = 2
        sus_tr = pairdf
        size_ratio = None
        rug_date = None
        lrgch_sus_tr = None

        def get_poolsize_compare(sus_tr,pairdf,pool_shrink_threshold):
            """ For each suspect swap event:
                Get max pool size before swap event and after swap event.
                Get ratio = size_after / size_before - if ratio below threshold X, disregard swap event"""

            sus_tr["poolsize_check"] = 0
            if not sus_tr.empty:
                all_sync_df = pairdf
                all_sync_df = all_sync_df.reset_index()
                all_sync_df = all_sync_df.drop("index",axis=1)
                weth_pos = sus_tr.reset_index()
                weth_pos = weth_pos["weth_pos"][0]

                def poolsize_compare(row,all_sync_df,weth_pos,pool_shrink_threshold):
                    """ Splits exchange contract sync events dataframe at a swap event.
                        Finds the average eth poolsize before and after the swap event.  """
                    resd = {}
                    ref_row = all_sync_df[all_sync_df["transaction_hash"] == row["transaction_hash"]]
                    ref_row = ref_row.reset_index()
                    poolsize_check = 0

                    #use current sample (row) as ref point
                    ref_aft = all_sync_df[all_sync_df["block_timestamp"] == (int(ref_row["block_timestamp"][0]))]
                    ref_id = ref_aft.index[0]

                    #split dataframe at reference point
                    resd["before_df"] = all_sync_df.iloc[:ref_id,:]
                    resd["after_df"]= all_sync_df.iloc[ref_id:,:]

                    if weth_pos == 1:
                        bef_list = resd["before_df"]["reserve0"].to_list()
                        aft_list = resd["after_df"]["reserve0"].to_list()
                    if weth_pos == 2:
                        bef_list = resd["before_df"]["reserve1"].to_list()
                        aft_list = resd["after_df"]["reserve1"].to_list()

                    bef_list = [abs(float(x)) for x in bef_list]
                    aft_list = [abs(float(x)) for x in aft_list]

                    try:
                        resd["size_before"] = sum(bef_list) / len(bef_list)
                    except (ValueError,ZeroDivisionError): #if no transactions exist in before list, set before size to 0
                        resd["size_before"] = 0
                    try:
                        resd["size_after"] = sum(aft_list) / len(aft_list)
                    except (ValueError,ZeroDivisionError): #if no transactions exist in after list, set after size to nan
                        resd["size_after"] = "not a number"

                    try:
                        size_ratio = resd["size_after"] / resd["size_before"]
                        #If max ETH poolsize after ref point is <X% of the max ETH poolsize before ref point, mark as rug

                        if size_ratio < pool_shrink_threshold:
                            poolsize_check = 1
                    except (ZeroDivisionError,TypeError): #if size before/after is 0
                        pass

                    return poolsize_check

            sus_tr["poolsize_check"] = sus_tr.apply(lambda x: poolsize_compare(x,all_sync_df,weth_pos,pool_shrink_threshold),axis=1)

            return sus_tr

        if isinstance(sus_tr, pd.DataFrame):
            if not sus_tr.empty:
                sus_tr = get_poolsize_compare(sus_tr,pairdf,pool_shrink_threshold)
                #select transactions where the ETH poolsize is and remains significantly smaller after current transaction
                sus_tr= sus_tr[sus_tr["poolsize_check"] == 1]

                #select the transaction where the largest drawdown in eth poolsize happened comapred to the previous sync event as the "rug-pull-transaction"
                #NOTE: this is not necessarily the precise transaction where the rug-pull itself occured.
                #This transaction is located close to a catastrophic drop in ETH poolsize the exchange pair never recovered from.
                if weth_pos == 1:
                    lrgch_sus_tr = sus_tr[sus_tr["deltaABS_res0"] == sus_tr["deltaABS_res0"].min()]
                if weth_pos == 2:
                    lrgch_sus_tr = sus_tr[sus_tr["deltaABS_res1"] == sus_tr["deltaABS_res1"].min()]
            else:
                lrgch_sus_tr = np.nan
        else:
            lrgch_sus_tr = np.nan

        if isinstance(lrgch_sus_tr, pd.DataFrame):
            if not lrgch_sus_tr.empty:
                #get size ratio & date of rug between sub-samples for record/storage
                lrgch_sus_tr_temp = lrgch_sus_tr.reset_index(drop=True)
                rug_date = lrgch_sus_tr_temp["block_timestamp"][0]

                ref_id = lrgch_sus_tr.index[0]

                resd = {}
                resd["before_df"] = pairdf.loc[:ref_id]
                resd["after_df"]= pairdf.loc[ref_id:]

                if weth_pos == 1:
                    bef_list = resd["before_df"]["reserve0"].to_list()
                    aft_list = resd["after_df"]["reserve0"].to_list()
                if weth_pos == 2:
                    bef_list = resd["before_df"]["reserve1"].to_list()
                    aft_list = resd["after_df"]["reserve1"].to_list()

                bef_list = [abs(float(x)) for x in bef_list]
                aft_list = [abs(float(x)) for x in aft_list]

                try:
                    resd["size_before"] = sum(bef_list) / len(bef_list)
                except (ValueError,ZeroDivisionError): #if no transactions exist in before list, set before size to 0
                    resd["size_before"] = 0
                try:
                    resd["size_after"] = sum(aft_list) / len(aft_list)
                except (ValueError,ZeroDivisionError): #if no transactions exist in after list, set after size to nan
                    resd["size_after"] = "not a number"

                try:
                    size_ratio = resd["size_after"] / resd["size_before"]
                except (ZeroDivisionError,TypeError): #if size before/after is 0
                    size_ratio = None

        return lrgch_sus_tr,size_ratio,rug_date

    ########################################################################################################
    ########################################################################################################

    def analyse_eoa(pair_df,analysis_res_dict,frac_losers_threshold):
        """ Analyses EOAs of a pair:
            Finds the net ETH balance of EOA in the recorded timeframe. Finds number of winners (pos balance) and losers (neg balance). """

        creator_balance = None
        #Get position of ETH pool
        if pair_df["weth_pos"][0] == 1:
            weth_pos = 1
        else:
            weth_pos = 2

        eoa_balance_dict = {}
        if weth_pos == 1: deltaABS = "deltaABS_res0"
        else: deltaABS = "deltaABS_res1"

        eoa_removed_eth_dict = {}
        tmpdf = pair_df[["from_address",deltaABS]].copy()
        resdf = tmpdf.groupby(["from_address"]).sum() #group by EOAs and get balance for each EOA
        unique_eoas_count = len(resdf.index)

        resdf[deltaABS] = resdf[deltaABS]*(-1)/1e18
        negbal = resdf[resdf[deltaABS] < 0]

        tmp = pair_df.reset_index(drop=True)
        creator_balance = resdf[resdf.index == tmp["creator"][0]]
        creator_balance = creator_balance.reset_index(drop=True)
        creator_balance = creator_balance[deltaABS][0]
        creator_balance = float(creator_balance) / 1e18

        frac_losers = len(negbal.index) / len(resdf.index) #get fraction of EOAs that has a net negative balance in time of recording
        #result dataframe
        eoa_df = pd.DataFrame()
        eoa_df["frac_losers_pair"] = frac_losers
        if frac_losers > frac_losers_threshold:
            high_losers_count = 1
        else:
            high_losers_count = 0


        eoa_analysis_results = {}
        eoa_analysis_results["eoa_df"] = eoa_df
        return high_losers_count,unique_eoas_count,frac_losers,creator_balance

    ########################################################################################################
    ########################################################################################################

    count = 0
    for ex_contr in targets:
        count += 1
        print("\n\nProcessign sample "+str(count))
        print("Working on contract: "+str(ex_contr))
        startTime = datetime.now()

        #Load trading pair from DataBase + preprocessing
        sync_df,short_sync_df,is_weth_pair,sync_events_count,pair_created_date,is_resampled,sync_events_count_resampled,resampling_decrease = group_sync_events_by_pair(ex_contr,db_path)
        size_ratio = None
        rug_time = None
        dur_until_rug = None
        eoa_results = None
        unique_eoas_count = None
        frac_losers = None
        creator_balance = None
        rug_confirmed = None
        low_syncs = None
        if sync_df is not None:
            if len(sync_df.index) > 10: #minimum number of transactions threshold
                low_syncs = 1
                print("Working on find_rug_tr() for contract: "+str(ex_contr))
                startTime = datetime.now()

                #Find rug pull indicating transaction
                transactions_analyzed,size_ratio,rug_time = find_rug_tr(short_sync_df,pool_shrink_threshold)
                if rug_time is not None:
                    dur_until_rug = rug_time - pair_created_date

                if transactions_analyzed is not None:
                    print("Working on analyse_eoa() for contract: "+str(ex_contr))
                    startTime = datetime.now()

                    #Analyse EOA balances
                    eoa_results,unique_eoas_count,frac_losers,creator_balance = analyse_eoa(sync_df,transactions_analyzed,frac_losers_threshold)

                else:
                    eoa_analysis_results_dict = None

            else:
                print("Contract "+str(ex_contr)+" has less than 10 transactions - disregarding..")
                low_syncs = 1
        else:
            print("Contract "+str(ex_contr)+" does not comply (returned as None Type) - disregarding..")

        print("Done with contract: "+str(ex_contr))
        eval_dict = {}
        eval_dict["pair_address"] = ex_contr
        eval_dict["pair_created_date"] = pair_created_date
        eval_dict["sync_events_count"] = sync_events_count #total number of sync events for pair
        eval_dict["is_weth_pair"] = is_weth_pair
        eval_dict["low_syncs"] = low_syncs
        eval_dict["rug_time"] = rug_time #timestamp of rug pull indicating transaction
        eval_dict["size_ratio"] = size_ratio #poolsize compare bef and aft rug
        if dur_until_rug is not None:
            dur_until_rug = ( float(dur_until_rug) / 60 ) / 60
        eval_dict["dur_until_rug_hours"] = dur_until_rug #duration until rug in hours - start = first sync event
        eval_dict["eoas_count"] = unique_eoas_count #unique eoas that interact with trading pair
        eval_dict["frac_losers"] = frac_losers #fraction EOAs with negative balance
        eval_dict["is_rugpull"] = rug_confirmed
        eval_dict["is_resampled"] = is_resampled
        eval_dict["sync_events_count_resampled"] = sync_events_count_resampled
        eval_dict["resampling_decrease"] = resampling_decrease

        eval_df = pd.DataFrame([eval_dict])

        csvFilePath = "(YOUR_PATH)\data\computer_generated\rugpull_identification_results\rugpull_identification_results.csv"
        if not os.path.isfile(csvFilePath):
            eval_df.to_csv(csvFilePath, mode='a', index=False)
        else:
            eval_df.to_csv(csvFilePath, mode='a', index=False, header=False)

    print("Done with eval for settings - pool_shrink_threshold = "+str(pool_shrink_threshold)+" ,frac_losers_threshold = "+str(frac_losers_threshold))

    return

def main():
    startTime = datetime.now()
    db_path = r"(YOUR_PATH)\data\base_dataset\uniswap_data.db"

    def connect(db_path):
        """ Connects to DataBase """
        conn = sqlite3.connect(db_path) #rc)
        cur = conn.cursor()
        return conn, cur

    #get all trading pairs in DB
    conn, cur = connect(db_path)
    query = " SELECT pair_address FROM pair "
    pair_df = pd.read_sql_query(query, conn)
    targets = pair_df["pair_address"].tolist()

    too_big_list = ["0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc","0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"] #USDC-WETH,USDT-WETH trading pairs excluded, because my machine cant load them
    too_big_list = [item.lower() for item in too_big_list]


    pool_shrink_threshold_input = [0.06] #liberal parameter configuration
    frac_losers_input = [0.46] #optimized parameter configuration

    mesh = np.array(np.meshgrid(pool_shrink_threshold_input,frac_losers_input)) #get all possible combinations of input params
    tmp_input_params = mesh.T.reshape(-1,2)
    tmp_input_params = tmp_input_params.tolist()

    input_params = []
    for input_param in tmp_input_params:
        input_param.append(targets)
        input_param.append(db_path)

        input_params.append(input_param)

    print("Total number of planned iterations: "+str(len(input_params)))
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(init_analysis,input_params)
    print("All Done - Time consumed: "+str(datetime.now() - startTime))
    print("Exiting..")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

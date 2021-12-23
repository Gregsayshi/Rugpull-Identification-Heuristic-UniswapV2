import os
import re
import json
import time
import sqlite3
import logging
import warnings
import statistics
import numpy as np
import pandas as pd
import pickle as pkl
import cProfile, pstats
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from statistics import StatisticsError

def init_analysis(pool_shrink_threshold,frac_losers_threshold,pos_targetarr,neg_targetarr,db_path):
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
        try:
            conn, cur = connect(db_path) # connect to the SQL server
            df_dict = {}

            try: #get pair table to see if pair is compliant to having ETH in one of the pools
                startTime = datetime.now()
                query = " SELECT * FROM pair WHERE pair_address = '%s' "
                records_to_insert = (target,)
                pair_df = pd.read_sql_query(query %records_to_insert, conn)

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
                read_sql_query_time = datetime.now() - startTime

                startTime = datetime.now()
                sync_df = get_dlt_sync(sync_df)
                get_dlt_sync_sql_query_time = datetime.now() - startTime
                sync_df["weth_pos"] = weth_pos
                creator_eoa = sync_df["from_address"][0] #first sync event = first transaction = initiated by contract creator eoa
                sync_df["creator"] = creator_eoa
                startTime = datetime.now()
                short_sync_df = shorten_sample(sync_df)
                shorten_sample_time = datetime.now() - startTime
                if len(sync_df.index) > len(short_sync_df.index):
                    print("Sync_df for contract "+str(ex_contr)+" shortend from "+str(len(sync_df.index))+" samples to "+str(len(short_sync_df.index))+" samples.")

            except Exception as e:
                print(f"Failed at group_sync_events_by_pair() with the following error:"+str(e))
        except Exception as e:
            print(f"Failed at establishing DB connect @ group_sync_events_by_pair() with the following error: "+str(e))
        finally:
            conn.commit()
            conn.close()
            return sync_df,short_sync_df,read_sql_query_time,get_dlt_sync_sql_query_time,shorten_sample_time

    ########################################################################################################
    ########################################################################################################

    # Analyse suspicious transactions
    def find_rug_tr(pairdf,pool_shrink_threshold):
        result_dict = {}
        if pairdf["weth_pos"][0] == 1:
            weth_pos = 1
        else:
            weth_pos = 2
        sus_tr = pairdf

        def get_poolsize_compare(sus_tr,pairdf,pool_shrink_threshold):
            """ For each suspect swap event.
                Get max pool size before ref point and after swap event.
                Get ratio = size_after / size_before - if ratio below threshold X, disregard swap event"""

            sus_tr["poolsize_check"] = 0
            if not sus_tr.empty:
                all_sync_df = pairdf
                all_sync_df = all_sync_df.reset_index()
                all_sync_df = all_sync_df.drop("index",axis=1)
                weth_pos = sus_tr.reset_index()
                weth_pos = weth_pos["weth_pos"][0]

                def poolsize_compare(row,all_sync_df,weth_pos,pool_shrink_threshold):
                    """ Splits exchange contract sync events dataframe at sync event.
                        Finds the maxmimum eth poolsize before and after sync event. """
                    resd = {}
                    ref_row = all_sync_df[all_sync_df["transaction_hash"] == row["transaction_hash"]]
                    ref_row = ref_row.reset_index()
                    poolsize_check = 0


                    #if no sample found after delay, use current sample (row) as ref point
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
                        #If max ETH poolsize after ref point is <10% of the max ETH poolsize before ref point, mark as rug
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
            result_dict["rug_tr"] = lrgch_sus_tr

        return result_dict

    ########################################################################################################
    ########################################################################################################

    def analyse_eoa(pair_df,analysis_res_dict,frac_losers_threshold):
        """ Analyses EOAs of a pair:
            Finds the net ETH balance of EOA in the recorded timeframe. Finds number of winners (pos balance) and losers (neg balance). """

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

        resdf[deltaABS] = resdf[deltaABS]*(-1)/1e18
        negbal = resdf[resdf[deltaABS] < 0]
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
        return high_losers_count

    ########################################################################################################
    ########################################################################################################

    def eval(eoa_results,suspect_results):
        """Evaluates results of heuristic"""

        if not suspect_results["rug_tr"].empty:
            if eoa_results is not None:
                if eoa_results == 1:
                    rug_confirmed = 1
                else:
                    rug_confirmed = 0
            else:
                rug_confirmed = 0
        else:
            rug_confirmed = 0

        return rug_confirmed

    ########################################################################################################
    ########################################################################################################

    TP = []
    TN = []
    FP = []
    FN = []
    targetarr = pos_targetarr + neg_targetarr
    group_sync_events_by_pair_time_arr = []
    find_rug_tr_time_arr = []
    group_by_eoa_time_arr = []
    eoa_analysis_results_dict_arr = []
    read_sql_query_time_arr = []
    get_dlt_sync_sql_query_time_arr = []
    shorten_sample_time_arr = []

    for ex_contr in targetarr:
        print("\n\nWorking on group_sync_events_by_pair() for contract: "+str(ex_contr))
        startTime = datetime.now()

        #Load trading pair from DataBase + preprocessing
        sync_df,short_sync_df,read_sql_query_time,get_dlt_sync_sql_query_time,shorten_sample_time = group_sync_events_by_pair(ex_contr,db_path)
        read_sql_query_time_arr.append(read_sql_query_time)
        get_dlt_sync_sql_query_time_arr.append(get_dlt_sync_sql_query_time)
        shorten_sample_time_arr.append(shorten_sample_time)
        group_sync_events_by_pair_time = datetime.now() - startTime
        group_sync_events_by_pair_time_arr.append(group_sync_events_by_pair_time)
        if sync_df is not None:
            if len(sync_df.index) > 10: #minimum number of transactions threshold
                print("Working on find_rug_tr() for contract: "+str(ex_contr))
                startTime = datetime.now()

                #Find rug pull indicating transaction
                transactions_analyzed = find_rug_tr(short_sync_df,pool_shrink_threshold)
                find_rug_tr_time = datetime.now() - startTime
                find_rug_tr_time_arr.append(find_rug_tr_time)
                if not transactions_analyzed["rug_tr"].empty:
                    print("Working on analyse_eoa() for contract: "+str(ex_contr))
                    startTime = datetime.now()

                    #Analyse EOA balances
                    eoa_analysis_results_dict = analyse_eoa(sync_df,transactions_analyzed,frac_losers_threshold)
                    eoa_analysis_results_dict_time = datetime.now() - startTime
                    eoa_analysis_results_dict_arr.append(eoa_analysis_results_dict_time)
                else:
                    eoa_analysis_results_dict = None
                rug_confirmed = eval(eoa_analysis_results_dict,transactions_analyzed)

                if rug_confirmed == 1:
                    if ex_contr in pos_targetarr:
                        TP.append(ex_contr)
                    if ex_contr in neg_targetarr:
                        FP.append(ex_contr)
                if rug_confirmed == 0:
                    if ex_contr in pos_targetarr:
                        FN.append(ex_contr)
                    if ex_contr in neg_targetarr:
                        TN.append(ex_contr)
            else:
                print("Contract "+str(ex_contr)+" has less than 10 transactions - disregarding..")
        else:
            print("Contract "+str(ex_contr)+" does not comply (returned as None Type) - disregarding..")
        print("Done with contract: "+str(ex_contr))

    try:
        accuracy = len(TP+TN) / len(TP+TN+FP+FN)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = len(TP) / len(TP+FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = len(TP) / len(TP+FN)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    eval_dict = {}
    eval_dict["pool_shrink_threshold"] = pool_shrink_threshold
    eval_dict["frac_losers_threshold"] = frac_losers_threshold
    eval_dict["accuracy"] = accuracy
    eval_dict["f1"] = f1
    eval_dict["TP"] = len(TP)
    eval_dict["TN"] = len(TN)
    eval_dict["FP"] = len(FP)
    eval_dict["FN"] = len(FN)
    eval_dict["TP_contracts"] = TP
    eval_dict["TN_contracts"] = TN
    eval_dict["FP_contracts"] = FP
    eval_dict["FN_contracts"] = FN
    eval_dict["Duration_group_sync_events"] = sum(group_sync_events_by_pair_time_arr,timedelta())
    eval_dict["Duration_find_rug_tr"] = sum(find_rug_tr_time_arr,timedelta())
    eval_dict["Duration_eoa_analysis"] = sum(eoa_analysis_results_dict_arr,timedelta())
    eval_dict["read_sql_query_time"] = sum(read_sql_query_time_arr,timedelta())
    eval_dict["get_dlt_sync_sql_query_time"] = sum(get_dlt_sync_sql_query_time_arr,timedelta())
    eval_dict["shorten_sample_time"] = sum(shorten_sample_time_arr,timedelta())
    eval_df = pd.DataFrame([eval_dict])

    csvFilePath = "(YOUR_PATH)\data\computer_generated\parameter_optimization_results\first_round_parameter_optimization_results.csv"

    if not os.path.isfile(csvFilePath):
        eval_df.to_csv(csvFilePath, mode='a', index=False)
    else:
        eval_df.to_csv(csvFilePath, mode='a', index=False, header=False)

    print("Done with eval for settings - pool_shrink_threshold = "+str(pool_shrink_threshold)+" ,frac_losers_threshold = "+str(frac_losers_threshold))

    return

def main():
    startTime = datetime.now()

    with open('(YOUR_PATH)\data\manually_generated\ground_truth_samples\Positive_Samples.txt') as f:
        pos_targetarr = [line.rstrip() for line in f]
        pos_targetarr = [line.lower() for line in pos_targetarr]

    with open('(YOUR_PATH)\data\manually_generated\ground_truth_samples\Negative_Samples.txt') as f:
        neg_targetarr = [line.rstrip() for line in f]
        neg_targetarr = [line.lower() for line in neg_targetarr]

    db_path = 	r"(YOUR_PATH)\data\base_dataset\uniswap_data.db"

    pool_shrink_threshold_input = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1]
    frac_losers_input = [0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6]

    mesh = np.array(np.meshgrid(pool_shrink_threshold_input,frac_losers_input)) #get all possible combinations of input params
    tmp_input_params = mesh.T.reshape(-1, 2)
    tmp_input_params = tmp_input_params.tolist()


    input_params = []
    for input_param in tmp_input_params:
        input_param.append(pos_targetarr)
        input_param.append(neg_targetarr)
        input_param.append(db_path)
        input_params.append(input_param)

    print("Total number of planned iterations: "+str(len(input_params)))
    pool = mp.Pool(mp.cpu_count()) #multi core processing
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

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
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("(YOUR_PATH)\data\computer_generated\parameter_optimization_results\first_round_parameter_optimization_results.csv",sep=",")
df_shrink = df.groupby(by=["pool_shrink_threshold"]).mean()
df_losers = df.groupby(by=["frac_losers_threshold"]).mean()
dfarr = ["shrink_df","losers_df"]

for dataframe in dfarr:
    if "losers_df" in dataframe:
        y = df_losers.f1
        x = df_losers.index
        xlabel = "Fraction EOAs with a negative base cryptoasset balance"
        ylabel = "Average F1 Score for given input parameter"
        title = "Average F1 score for frac_losers_threshold input parameter values"
        filename = "frac_losers.pdf"
        color = "red"
    if "shrink_df" in dataframe:
        y = df_shrink.f1
        x = df_shrink.index
        ylabel = "Average F1 Score for given input parameter"
        xlabel = "Maximum base cryptoasset poolsize after rug-pull as a fraction of maximum poolsize before the rug-pull"
        title = "Average F1 score for pool_shrink_threshold input parameter values"
        filename = "shrink.pdf"
        color = "blue"

    figure = plt.figure(figsize=(32,16))
    ax = figure.add_subplot(111)
    ax.plot(x,y,color = color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig = ax.get_figure()
    print(fig)
    fig.savefig(r"(YOUR_PATH)\data\computer_generated\figures\parameter_optimization\initial_round_"+filename)
    plt.close()


df = pd.read_csv("(YOUR_PATH)\data\computer_generated\parameter_optimization_results\second_round_parameter_optimization_results.csv",sep=",")
df_shrink = df.groupby(by=["pool_shrink_threshold"]).mean()
df_losers = df.groupby(by=["frac_losers_threshold"]).mean()
dfarr = ["shrink_df","losers_df"]

for dataframe in dfarr:
    if "losers_df" in dataframe:
        y = df_losers.f1
        x = df_losers.index
        xlabel = "Fraction EOAs with a negative base cryptoasset balance"
        ylabel = "Average F1 Score for given input parameter"
        title = "Average F1 score for frac_losers_threshold input parameter values"
        filename = "frac_losers.pdf"
        color = "red"
    if "shrink_df" in dataframe:
        y = df_shrink.f1
        x = df_shrink.index
        ylabel = "Average F1 Score for given input parameter"
        xlabel = "Maximum base cryptoasset poolsize after rug-pull as a fraction of maximum poolsize before the rug-pull"
        title = "Average F1 score for pool_shrink_threshold input parameter values"
        filename = "shrink.pdf"
        color = "blue"

    figure = plt.figure(figsize=(32,16))
    ax = figure.add_subplot(111)
    ax.plot(x,y,color = color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig = ax.get_figure()
    print(fig)
    fig.savefig(r"(YOUR_PATH)\data\computer_generated\figures\parameter_optimization\second_round_"+filename)
    plt.close()

print("ALL DONE.")

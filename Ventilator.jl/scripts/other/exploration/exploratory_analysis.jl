using DrWatson
@quickactivate

using Ventilator
using DataFrames, CSV
using Plots, StatsPlots

df = CSV.load(datadir("train.csv"), DataFrame)
groups = groupby(df, :breath_id)

###############################################################
### Overall distribution of data & full dataset exploration ###
###############################################################

describe(df)
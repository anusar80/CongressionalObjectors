#%% This code replicates the results of "Are Congressmen who challenged the election results in thrall of Trump's base?"
# published on the PolicyTensor.com on Jan 13, 2021. 

#%% Import modules
from math import sqrt
import os
# import dask
# os.environ["MODIN_ENGINE"] = "dask"  # dask or ray
import pandas as pd
import numpy as np
import quandl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa import seasonal
from statsmodels.tsa import filters
from scipy import signal
from statsmodels.tsa.seasonal import STL
# from stldecompose import decompose, forecast
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from statsmodels.api import Logit
from pingouin import corr, partial_corr
import xlsxwriter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, distance
from scipy.cluster import hierarchy
from scipy.stats import ttest_ind as ttest
from sklearn.decomposition import PCA
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import norm, skewnorm, skew
from matplotlib import dates as mdates
from sklearn.decomposition import PCA
from pyppca import ppca
import plotly.graph_objects as go
import warnings
import time
from beepy import beep
from datetime import date, datetime
from matplotlib.dates import DateFormatter
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
sns.set()
#%% To fix the date fuck-ups
old_epoch = '0000-12-31T00:00:00'
new_epoch = '1970-01-01T00:00:00'
mdates.set_epoch(old_epoch)
plt.rcParams['date.epoch'] = '000-12-31'
plt.rcParams['axes.facecolor'] = 'w'
plt.style.use('seaborn')
register_matplotlib_converters()
#%% Clock utility function
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti
TicToc = TicTocGenerator()
def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)
def tic():
    toc(False)

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, ':')

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

#%% Get international index returns and vol data
os.chdir('/Users/anusarfarooqui/Docs/Matlab/IPUMS21')
# Congressional district and county https://www.census.gov/geographies/reference-files/2010/geo/relationship-files.html
url = 'https://www2.census.gov/geo/relfiles/cdsld18/natl/natl_cocd_delim.txt'
CD_to_county_fips = pd.read_csv(url, header=1)
state_fips_iso = pd.read_excel('state_fips.xlsx', header=0)
print(state_fips_iso)
CD_to_county = CD_to_county_fips.merge(state_fips_iso, how='left', left_on='State', right_on='FIPS')
CD_to_county['County_Key'] = CD_to_county['State'].astype('str') + '-' + CD_to_county['County'].astype('str')
CD_to_county['CD'] = CD_to_county['Code'].astype('str') + '-' + CD_to_county['Congressional District'].astype('str')
print(CD_to_county)
Objectors = pd.read_excel('Objectors.xlsx', header=0)
print(Objectors)
IPUMS_fmt = pd.read_excel('IPUMS_fmt.xlsx', header=0)
print(IPUMS_fmt)

#%% Read the giant IPUMS file
# tic()
# IPUMS = pd.read_fwf(filepath_or_buffer='usa_00011.dat', widths=IPUMS_fmt['Len'], names=IPUMS_fmt.Variable, engine='c')
# toc()
# print(IPUMS.head())
# IPUMS.to_pickle('IPUMS.pkl') # Saved to pickle — Read from there
#%% Merge with CD_to_county to get districts
IPUMS = pd.read_pickle('IPUMS.pkl')
IPUMS.County_Key.unique()
df = IPUMS.merge(CD_to_county, how='left', left_on=['County_Key'], right_on=['County_Key'], suffixes=['_IPUMS', '_Census'])
#%% Define socioeconomic variables
data = pd.DataFrame({'HHIncome': df.HHINCOME.groupby(df.CD).median() / 1000,
                     'SEI': df.FOODSTMP.groupby(df.CD).mean(),
                     'Poverty': df.POVERTY.groupby(df.CD).median(),
                     'Prestige': df.PRESGL.groupby(df.CD).mean(),
                     'EDUC': df.EDUC.groupby(df.CD).mean(),
                     'White_HHIncome': df.HHINCOME[np.logical_and(df.RACE==1, df.HISPAN==0)].groupby(
                         df.CD[np.logical_and(df.RACE==1, df.HISPAN==0)]).median() / 1000,
                     'Black_HHIncome': df.HHINCOME[np.logical_and(df.RACE==2, df.HISPAN==0)].groupby(
                         df.CD[np.logical_and(df.RACE==2, df.HISPAN==0)]).median() / 1000,
                     'Hispanic_HHIncome': df.HHINCOME[df.HISPAN>0].groupby(df.CD[df.HISPAN>0]).median() / 1000,
                     'White_SEI': df.SEI[np.logical_and(df.RACE == 1, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 1, df.HISPAN == 0)]).mean(),
                     'Black_SEI': df.SEI[np.logical_and(df.RACE == 2, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 2, df.HISPAN == 0)]).mean(),
                     'Hispanic_SEI': df.SEI[df.HISPAN > 0].groupby(df.CD[df.HISPAN > 0]).mean(),
                     'White_Poverty': df.POVERTY[np.logical_and(df.RACE == 1, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 1, df.HISPAN == 0)]).median(),
                     'Black_Poverty': df.POVERTY[np.logical_and(df.RACE == 2, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 2, df.HISPAN == 0)]).median(),
                     'Hispanic_Poverty': df.POVERTY[df.HISPAN > 0].groupby(df.CD[df.HISPAN > 0]).median(),
                     'White_Prestige': df.PRESGL[np.logical_and(df.RACE == 1, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 1, df.HISPAN == 0)]).mean(),
                     'Black_Prestige': df.PRESGL[np.logical_and(df.RACE == 2, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 2, df.HISPAN == 0)]).mean(),
                     'Hispanic_Prestige': df.PRESGL[df.HISPAN > 0].groupby(df.CD[df.HISPAN > 0]).mean(),
                     'White_EDUC': df.EDUC[np.logical_and(df.RACE == 1, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 1, df.HISPAN == 0)]).mean(),
                     'Black_EDUC': df.EDUC[np.logical_and(df.RACE == 2, df.HISPAN == 0)].groupby(
                         df.CD[np.logical_and(df.RACE == 2, df.HISPAN == 0)]).mean(),
                     'Hispanic_EDUC': df.EDUC[df.HISPAN > 0].groupby(df.CD[df.HISPAN > 0]).mean(),
                     })
data = Objectors.merge(data, how='left', left_on='District', right_on='CD')
#%% T tests: race blind
criteria = ['EDUC', 'HHIncome', 'Prestige', 'Poverty']

just_obj_tStat = np.zeros(len(criteria))
just_obj_pVal = np.zeros(len(criteria))
just_obj_mean = np.zeros(len(criteria))
just_not_obj_mean = np.zeros(len(criteria))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][data.Objected==1],
                   data[criteria[i]][data.Objected==0], equal_var=False, nan_policy='omit')
    just_obj_mean[i] = data[criteria[i]][data.Objected==1].mean()
    just_not_obj_mean[i] = data[criteria[i]][data.Objected == 0].mean()
    just_obj_tStat[i] = result[0]
    just_obj_pVal[i] = result[1]

just_obj = pd.DataFrame({'Objected': just_obj_mean, 'No_Objection': just_not_obj_mean,
                         'tStat': just_obj_tStat, 'pVal': just_obj_pVal}, index=criteria)
print(just_obj)

party_tStat = np.zeros(len(criteria))
party_pVal = np.zeros(len(criteria))
party_GOP = np.zeros(len(criteria))
party_DEM = np.zeros(len(criteria))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][data.Party=='Republican'],
               data[criteria[i]][data.Party=='Democrat'], equal_var=False, nan_policy='omit')
    party_GOP[i] = data[criteria[i]][data.Party=='Republican'].mean()
    party_DEM[i] = data[criteria[i]][data.Party == 'Democrat'].mean()
    party_tStat[i] = result[0]
    party_pVal[i] = result[1]

parties = pd.DataFrame({'GOP': party_GOP, 'DEM': party_DEM, 'tStat': party_tStat, 'pVal': party_pVal}, index=criteria)
print(parties)

objector_tStat = np.zeros(len(criteria))
objector_pVal = np.zeros(len(criteria))
objector_mean = np.zeros(len(criteria))
non_objector_mean = np.zeros((len(criteria)))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][np.logical_and(data.Objected==1, data.Party=='Republican')],
                   data[criteria[i]][np.logical_and(data.Objected==0, data.Party=='Republican')],
                   equal_var=False, nan_policy='omit')
    objector_mean[i] = data[criteria[i]][np.logical_and(data.Objected==1, data.Party=='Republican')].mean()
    non_objector_mean[i] = data[criteria[i]][np.logical_and(data.Objected==0, data.Party=='Republican')].mean()
    objector_tStat[i] = result[0]
    objector_pVal[i] = result[1]

objection = pd.DataFrame({'Objector': objector_mean, 'No_Objection': non_objector_mean,
                          'tStat': objector_tStat, 'pVal': objector_pVal}, index=criteria)
print(objection)

#%% T tests: race-specific
criteria = ['White_EDUC', 'White_HHIncome', 'White_Prestige', 'White_Poverty']

just_obj_tStat = np.zeros(len(criteria))
just_obj_pVal = np.zeros(len(criteria))
just_obj_mean = np.zeros(len(criteria))
just_not_obj_mean = np.zeros(len(criteria))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][data.Objected==1],
                   data[criteria[i]][data.Objected==0], equal_var=False, nan_policy='omit')
    just_obj_mean[i] = data[criteria[i]][data.Objected==1].mean()
    just_not_obj_mean[i] = data[criteria[i]][data.Objected == 0].mean()
    just_obj_tStat[i] = result[0]
    just_obj_pVal[i] = result[1]

just_obj = pd.DataFrame({'Objected': just_obj_mean, 'No_Objection': just_not_obj_mean,
                         'tStat': just_obj_tStat, 'pVal': just_obj_pVal}, index=criteria)
print(just_obj)

party_tStat = np.zeros(len(criteria))
party_pVal = np.zeros(len(criteria))
party_GOP = np.zeros(len(criteria))
party_DEM = np.zeros(len(criteria))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][data.Party=='Republican'],
               data[criteria[i]][data.Party=='Democrat'], equal_var=False, nan_policy='omit')
    party_GOP[i] = data[criteria[i]][data.Party=='Republican'].mean()
    party_DEM[i] = data[criteria[i]][data.Party == 'Democrat'].mean()
    party_tStat[i] = result[0]
    party_pVal[i] = result[1]

parties = pd.DataFrame({'GOP': party_GOP, 'DEM': party_DEM, 'tStat': party_tStat, 'pVal': party_pVal}, index=criteria)
print(parties)

objector_tStat = np.zeros(len(criteria))
objector_pVal = np.zeros(len(criteria))
objector_mean = np.zeros(len(criteria))
non_objector_mean = np.zeros((len(criteria)))

for i in range(len(criteria)):
    result = ttest(data[criteria[i]][np.logical_and(data.Objected==1, data.Party=='Republican')],
                   data[criteria[i]][np.logical_and(data.Objected==0, data.Party=='Republican')],
                   equal_var=False, nan_policy='omit')
    objector_mean[i] = data[criteria[i]][np.logical_and(data.Objected==1, data.Party=='Republican')].mean()
    non_objector_mean[i] = data[criteria[i]][np.logical_and(data.Objected==0, data.Party=='Republican')].mean()
    objector_tStat[i] = result[0]
    objector_pVal[i] = result[1]

objection = pd.DataFrame({'Objector': objector_mean, 'No_Objection': non_objector_mean,
                          'tStat': objector_tStat, 'pVal': objector_pVal}, index=criteria)
print(objection)
#%% Box plots
criteria = np.array(['EDUC', 'HHIncome', 'Prestige', 'Poverty'])
for i in range(len(criteria)):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, sharex=True, dpi=400)
    sns.boxplot(x='Objected', y=criteria[i], hue='Party', hue_order=['Democrat', 'Republican'],
                data=data, palette='colorblind', ax=ax[0])
    sns.boxplot(x='Objected', y='White_' + criteria[i], hue='Party', hue_order=['Democrat', 'Republican'],
                data=data, palette='colorblind', ax=ax[1])
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.5, wspace=0.3)
    plt.savefig('box_' + criteria[i] + '.png')
    plt.show()

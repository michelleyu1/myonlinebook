#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ulmo   # fetch snotel
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import geopandas as gpd
from shapely import wkt
import contextily as ctx
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import shape, Point, Polygon, box


# In[ ]:


import statistics


# In[ ]:


# start_date = datetime(1999,10,1)
# end_date = datetime(2004,9,30)
start_date = datetime(1985,10,1)
end_date = datetime(2016,9,30)


# In[ ]:


# pr_prism_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_codebase_reproduction/v8/swe_part_pr_lag1_fltr_2_snw_szn_by_snw_acc_ok/'
# experiments_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/'
pr_prism_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/prism/'
experiments_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/'
save_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/analysis/bias_decomposition/'


# In[ ]:


# Peak SWE
# pr_gridmet_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/pr_gridmet/'
# pr_prism_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/temp_gridmet/'
# pr_gridmet_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/pr_gridmet_temp_gridmet/'
# pr_livneh_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/pr_livneh21/'
# pr_prism_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/temp_livneh21/'
# pr_livneh_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/pr_livneh21_temp_livneh21/'
# th_jennings_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/2000_to_2005/th_jennings/'
pr_gridmet_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/pr_gridmet/'
pr_prism_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/temp_gridmet/'
pr_gridmet_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/pr_gridmet_temp_gridmet/'
pr_livneh_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/pr_livneh21/'
pr_prism_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/temp_livneh21/'
pr_livneh_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/pr_livneh21_temp_livneh21/'
th_jennings_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/th_jennings/'


# In[ ]:


# Accumulation SWE
pr_gridmet_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_pr_gridmet/'
pr_prism_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_temp_gridmet/'
pr_gridmet_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_pr_gridmet_temp_gridmet/'
pr_livneh_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_pr_livneh21/'
pr_prism_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_temp_livneh21/'
pr_livneh_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/acc_pr_livneh21_temp_livneh21/'
th_jennings_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/th_jennings/'


# In[ ]:


# Ablation SWE
pr_gridmet_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_pr_gridmet/'
pr_prism_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_temp_gridmet/'
pr_gridmet_temp_gridmet_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_pr_temp_gridmet/'
pr_livneh_temp_prism_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_pr_livneh21/'
pr_prism_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_temp_livneh21/'
pr_livneh_temp_livneh_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/abl_pr_livneh21_temp_livneh21/'
th_jennings_dir = '/global/cscratch1/sd/yum/swe/UofA_bias_decomposition/experiments/1985_to_2016/prism/'


# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years
#       average peak SWE across years
#    average peak SWE across stations
#    calculate difference (compared to SNOTEL baseline)


# In[ ]:





# In[ ]:


# Load list of SNOTEL sites with available data
sites_df = pd.read_csv("sites_df.csv", index_col='Unnamed: 0')


# In[ ]:


sites_df['geometry'] = sites_df['geometry'].apply(wkt.loads)


# In[ ]:


# Convert to a Geopandas gdf
sites_gdf = gpd.GeoDataFrame(sites_df, crs='EPSG:4326')


# In[ ]:





# In[ ]:


# Get shapefile for Upper Colorado Riber Basin (UCRB)
uc_shp = "/global/cscratch1/sd/yum/swe/Upper_Colorado_River_Basin_Boundary/Upper_Colorado_River_Basin_Boundary.shp"

# Read UCRB shapefile
gm_poly_gdf = gpd.read_file(uc_shp, encoding="utf-8")

# Get bounds of UCRB
gm_poly_geom = gm_poly_gdf.iloc[0].geometry

# Determine sites in UCRB
sites_idx = sites_gdf.intersects(gm_poly_geom)

# Subset df to sites in UCRB
gm_snotel_sites = sites_gdf.loc[sites_idx]


# In[ ]:





# # Calculate bias of each peak SWE; proportions - 08/19/22

# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       calculate difference in peak SWE (compared to SNOTEL baseline)
# boxplot of peak SWE differences 


# ## Compute peak SWE (for each station and year)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


# initialize lists
site_code = []
year = []
pswe_date_snotel, pswe_snotel = [], []
pswe_date_ua, pswe_ua = [], [] 
pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)
            


# ## Compute Biases/Differences

# In[ ]:


# UA difference
ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel))


# ### Level 1

# In[ ]:


snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel))


# ### Level 2

# In[ ]:


prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# ### Level 3

# #### 3ai

# ##### Gridmet

# In[ ]:


pgridmet_bias = (np.array(pswe_pgridmet_tprism) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# ##### Livneh

# In[ ]:


plivneh_bias = (np.array(pswe_plivneh_tprism) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# #### 3aii

# ##### Gridmet

# In[ ]:


tgridmet_bias = (np.array(pswe_pprism_tgridmet) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# ##### Livneh

# In[ ]:


tlivneh_bias = (np.array(pswe_pprism_tlivneh) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# #### 3aiii

# ##### Gridmet

# In[ ]:


ptgridmet_bias = (np.array(pswe_pgridmet_tgridmet) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# ##### Livneh

# In[ ]:


ptlivneh_bias = (np.array(pswe_plivneh_tlivneh) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# #### 3b

# In[ ]:


thjennings_bias = (np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_pprism_tprism)) * (np.array(pswe_pprism_tprism) / np.array(pswe_snotel))


# In[ ]:





# ## Plots

# ### Boxplot of all peak SWE differences

# In[ ]:


pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])


# In[ ]:


# convert to %
pswe_diff_df = pswe_diff_df*100


# In[ ]:


# flip order of runs (i.e. columns)
pswe_diff_df = pswe_diff_df.iloc[:, ::-1]


# In[ ]:


# colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

fig,ax = plt.subplots(figsize=(10,6)) 
plt.axvline(100, color='black')
ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)
plt.grid()
plt.xlim(0,200)
plt.xlabel('Peak SWE difference')
plt.ylabel('Run')
plt.title('Peak SWE Error')
plt.savefig(save_dir+'boxplot_peak_swe_error_proportions', dpi=300)
# plt.show()


# ### Barplot of Mean Peak SWE

# In[ ]:


pswe_mean_diff_list = list(pswe_diff_df.mean())


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:





# In[ ]:


x = pswe_mean_diff_df.index.values
y_bot = [100, 100, 100, 100, 100, 100, 100, 100, 100]
y_dif = [pswe_mean_diff_df['peak_swe_diff'][i]-100 for i in range(9)]  

lbls = [f'{round(item, 2)}%' for item in y_dif]

plt.figure(figsize=(8,5))
bars = plt.barh(x, width=y_dif, height=0.6, left=y_bot)
# plt.bar_label(bars, fmt='%.2f')
plt.bar_label(bars, labels=lbls)
plt.title('Peak SWE error')
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.axvline(100, color='k', linestyle='--')
for color, bar in zip(['red','palevioletred','palevioletred','palevioletred','lightsalmon','lightsalmon','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(0,200)
plt.savefig(save_dir+'boxplot_mean_peak_swe_error_proportions', dpi=300)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# In[ ]:


ax = (pswe_mean_diff_df-100).plot(kind='barh', figsize=(8,5))
ax.bar_label(ax.containers[0], fmt='%.2f')
ax.set_xlim(-100,100)
ax.axvline(0, color='black')
ax.set_title('Peak SWE bias')
ax.set_xlabel('Peak SWE difference (%)')
ax.set_ylabel('Run')
for color, bar in zip(['red','palevioletred','palevioletred','palevioletred','lightsalmon','lightsalmon','lightsalmon','darkred','blue'], ax.patches):
    bar.set_color(color)
# plt.savefig(save_dir+'barplot_mean_peak_swe_bias.png', dpi=300)


# In[ ]:





# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       calculate difference in peak SWE (compared to SNOTEL baseline)
# boxplot of peak SWE differences 


# In[ ]:





# # Compute peak SWE (for each station and year) by function call

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


def generate_boxplot():
    


# In[ ]:


def mean_error_decomp_plot():
    


# In[ ]:


def median_error_decomp_plot():
    


# In[ ]:


def get_error_decomposition_plots(pr_prism_temp_prism_dir, pr_gridmet_temp_prism_dir, pr_prism_temp_gridmet_dir, pr_gridmet_temp_gridmet_dir, pr_livneh_temp_prism_dir, pr_prism_temp_livneh_dir, th_jennings_dir):
    # Initialize lists
    site_code = []
    year = []
    pswe_date_snotel, pswe_snotel = [], []
    pswe_date_ua, pswe_ua = [], [] 
    pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
    pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
    pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
    pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
    pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
    pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
    pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
    pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []
    
    # Get peak SWE value for each run
    for idx, row in gm_snotel_sites.iterrows():
        sitecode = row['code']
        # print(sitecode)
        # site_lon, site_lat = row['geometry'].x, row['geometry'].y

        if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
            # Load data
            pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
            pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')

            for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
                # print(yr)
                site_code.append(sitecode)
                year.append(yr)
                pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
                pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
                pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
                pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
                pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
                pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
                pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
                pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
                pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
                pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)

    # Compute errors by level
    ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel)) - 1
    snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel)) - 1
    prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1
    pgridmet_bias = ((np.array(pswe_pgridmet_tprism) / np.array(pswe_pprism_tprism)) - 1)
    plivneh_bias = ((np.array(pswe_plivneh_tprism) / np.array(pswe_pprism_tprism)) - 1)
    tgridmet_bias = ((np.array(pswe_pprism_tgridmet) / np.array(pswe_pprism_tprism)) - 1)
    tlivneh_bias = ((np.array(pswe_pprism_tlivneh) / np.array(pswe_pprism_tprism)) - 1)
    ptgridmet_bias = ((np.array(pswe_pgridmet_tgridmet) / np.array(pswe_pprism_tprism)) - 1)
    ptlivneh_bias = ((np.array(pswe_plivneh_tlivneh) / np.array(pswe_pprism_tprism)) - 1)
    thjennings_bias = ((np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_pprism_tprism)) - 1)
    
    # Generate Boxplot
    pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])
    pswe_diff_df = pswe_diff_df*100   # convert to %
    pswe_diff_df = pswe_diff_df.iloc[:, ::-1]       # flip order of runs (i.e. columns)
    #plot
    # colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
    colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

    fig,ax = plt.subplots(figsize=(10,6)) 
    plt.axvline(0, color='black')
    ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
    for patch,color in zip(props['boxes'],colors):
        patch.set_facecolor(color)
    plt.grid()
    plt.xlim(-100,100)
    plt.xlabel('Peak SWE difference (%)')
    plt.ylabel('Run')
    plt.title('Peak SWE Bias')
    # plt.savefig(save_dir+'boxplot_peak_swe_bias', dpi=300)
    # plt.show()
    
    # Generate error decomposition plot - mean
    lb_q, ub_q = 0.25, 0.75
    pswe_mean_diff_list = list(pswe_diff_df.mean())
    pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                     index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                     columns = ['peak_swe_diff'])
    prism_mean_peak_swe_bias = pswe_mean_diff_df['peak_swe_diff']['PRISM']
    #plot
    x = pswe_mean_diff_df.index.values
    y_bot = [prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, 0, 0]
    y_dif = [pswe_mean_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_mean_peak_swe_bias,0]

    lbls = [f'{round(item, 2)}%' for item in y_dif]

    # creating error
    lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
    lb_diff = pswe_mean_diff_df - lb
    ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
    ub_diff = ub - pswe_mean_diff_df
    xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(16,8))
    # bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
    h = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.85, 1]
    hnew = [i for i in h]
    bars = plt.barh(x, y_dif, xerr=xerror, height=hnew, left=y_bot, ecolor='k', error_kw=dict(lw=2, capsize=4, capthick=2))
    # plt.bar_label(bars, labels=lbls)    # label bars
    plt.title('Mean Peak SWE Error', fontsize=24)
    plt.xlabel('Peak SWE Difference (%)', fontsize=20)
    # plt.ylabel('Run')
    plt.axvline(0, color='k', linestyle='--', linewidth=2.5)
    plt.axvline(prism_mean_peak_swe_bias, color='grey', alpha=1.0, linestyle='--', linewidth=2.5)
    for color, bar in zip(['blue','deepskyblue','deepskyblue','palevioletred','skyblue','skyblue','lightsalmon','darkred','blue'], bars.patches):
        bar.set_color(color)
    plt.xlim(-50,50)

    # Working with axes
    for d in ["left", "top", "right"]:   # set plot spines to invisible
        plt.gca().spines[d].set_visible(False)
    plt.tick_params(axis='y', left=False)   # remove tick marks, keep tick labels

    # Text size/fonts
    # f = [16, 16, 16, 16, 16, 16, 16, 18, 20]
    # fnew = [i for i in f]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tick_params(axis="y",direction="in", pad=-25)

    # plt.savefig('/global/cscratch1/sd/yum/swe/conferences/MtnClim/'+'mean_peak_swe_error', dpi=300)
    
    
    
    # Generate error decomposition plot - median
    lb_q, ub_q = 0.25, 0.75
    pswe_med_diff_list = list(pswe_diff_df.median())
    pswe_med_diff_df = pd.DataFrame(pswe_med_diff_list, 
                                    index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                    columns = ['peak_swe_diff'])
    prism_med_peak_swe_bias = pswe_med_diff_df['peak_swe_diff']['PRISM']
    #plot
    x = pswe_med_diff_df.index.values
    y_bot = [prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, 0, 0]
    y_dif = [pswe_med_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_med_peak_swe_bias,0]

    lbls = [f'{round(item, 2)}%' for item in y_dif]

    # creating error
    lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
    lb_diff = pswe_med_diff_df - lb
    ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
    ub_diff = ub - pswe_med_diff_df
    xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

    # plt.figure(figsize=(8,5))
    plt.figure(figsize=(16,8))
    # bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
    h = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.85, 1]
    hnew = [i for i in h]
    bars = plt.barh(x, y_dif, xerr=xerror, height=hnew, left=y_bot, ecolor='k', error_kw=dict(lw=2, capsize=4, capthick=2))
    # plt.bar_label(bars, labels=lbls)    # label bars
    plt.title('Median Peak SWE Error', fontsize=24)
    plt.xlabel('Median SWE Difference (%)', fontsize=20)
    # plt.ylabel('Run')
    plt.axvline(0, color='k', linestyle='--', linewidth=2.5)
    plt.axvline(prism_med_peak_swe_bias, color='grey', alpha=1.0, linestyle='--', linewidth=2.5)
    for color, bar in zip(['blue','deepskyblue','deepskyblue','palevioletred','skyblue','skyblue','lightsalmon','darkred','blue'], bars.patches):
        bar.set_color(color)
    plt.xlim(-50,50)

    # working with axes
    for d in ["left", "top", "right"]:   # set plot spines to invisible
        plt.gca().spines[d].set_visible(False)
    plt.tick_params(axis='y', left=False)   # remove tick marks, keep tick labels

    # text size/fonts
    # f = [16, 16, 16, 16, 16, 16, 16, 18, 20]
    # fnew = [i for i in f]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tick_params(axis="y",direction="in", pad=-25)

    # plt.savefig('/global/cscratch1/sd/yum/swe/conferences/MtnClim/'+'med_peak_swe_error', dpi=300)

    return pswe_diff_df


# In[ ]:


an = get_error_decomposition_plots(pr_prism_temp_prism_dir, pr_gridmet_temp_prism_dir, pr_prism_temp_gridmet_dir, pr_gridmet_temp_gridmet_dir, pr_livneh_temp_prism_dir, pr_prism_temp_livneh_dir, th_jennings_dir)


# In[ ]:


an


# In[ ]:





# # Compute peak SWE (for each station and year)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


# initialize lists
site_code = []
year = []
pswe_date_snotel, pswe_snotel = [], []
pswe_date_ua, pswe_ua = [], [] 
pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        # pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        # pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            # pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            # pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)
            


# ## Compute Biases/Differences

# In[ ]:


# UA difference
ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel)) - 1


# ### Level 1

# In[ ]:


snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel)) - 1


# ### Level 2

# In[ ]:


prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1


# ### Level 3

# #### 3ai

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


pgridmet_bias = ((np.array(pswe_pgridmet_tprism) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


plivneh_bias = ((np.array(pswe_plivneh_tprism) / np.array(pswe_pprism_tprism)) - 1)


# #### 3aii

# ##### Gridmet

# In[ ]:


# (pswe_pprism_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


tgridmet_bias = ((np.array(pswe_pprism_tgridmet) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_pprism_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


tlivneh_bias = ((np.array(pswe_pprism_tlivneh) / np.array(pswe_pprism_tprism)) - 1)


# #### 3aiii

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


ptgridmet_bias = ((np.array(pswe_pgridmet_tgridmet) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


ptlivneh_bias = ((np.array(pswe_plivneh_tlivneh) / np.array(pswe_pprism_tprism)) - 1)


# #### 3b

# In[ ]:


# (pswe_pprism_tprism_thjennings_smean / pswe_snotel_smean) - 1


# In[ ]:


thjennings_bias = ((np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_pprism_tprism)) - 1)


# In[ ]:





# ## Plots

# In[ ]:


pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])


# In[ ]:


# convert to %
pswe_diff_df = pswe_diff_df*100


# In[ ]:


# flip order of runs (i.e. columns)
pswe_diff_df = pswe_diff_df.iloc[:, ::-1]


# ### Boxplot of all peak SWE differences

# In[ ]:


# Peak SWE
# colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

fig,ax = plt.subplots(figsize=(10,6)) 
plt.axvline(0, color='black')
ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)
plt.grid()
plt.xlim(-100,100)
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.title('Peak SWE Bias')
# plt.savefig(save_dir+'boxplot_peak_swe_bias', dpi=300)
# plt.show()


# ### Barplot of Mean Peak SWE

# In[ ]:


lb_q, ub_q = 0.25, 0.75


# In[ ]:


pswe_mean_diff_list = list(pswe_diff_df.mean())


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y_bot = np.linspace(30, 50, 10)
y_dif = np.linspace(10, 5, 10)

plt.barh(x, y_dif, left=y_bot)


# In[ ]:


prism_mean_peak_swe_bias = pswe_mean_diff_df['peak_swe_diff']['PRISM']


# In[ ]:


x = pswe_mean_diff_df.index.values
y_bot = [prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, 0, 0]
y_dif = [pswe_mean_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_mean_peak_swe_bias,0]
# y_bot = [pswe_mean_diff_df['peak_swe_diff'][i] + prism_mean_peak_swe_bias for i in range(7)] + [0,0]
# y_dif = [pswe_mean_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_mean_peak_swe_bias,0]
# y_bot = [pswe_mean_diff_df['peak_swe_diff'][i] + prism_mean_peak_swe_bias for i in range(7)] + [0,0]
# y_dif = [prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, 0]

lbls = [f'{round(item, 2)}%' for item in y_dif]

# creating error
lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
lb_diff = pswe_mean_diff_df - lb
ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
ub_diff = ub - pswe_mean_diff_df
xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

plt.figure(figsize=(8,5))
# bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
bars = plt.barh(x, y_dif, xerr=xerror, height=0.8, left=y_bot, ecolor='k', error_kw=dict(lw=1.25, capsize=2.5, capthick=1.25))
# plt.bar_label(bars, fmt='%.2f')
plt.bar_label(bars, labels=lbls)
plt.title('Mean peak SWE error')
plt.xlabel('Peak SWE difference (%)')
# plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(prism_mean_peak_swe_bias, color='grey', alpha=0.75, linestyle='--')
for color, bar in zip(['red','deepskyblue','deepskyblue','palevioletred','skyblue','lightsalmon','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-75,75)

# plt.savefig(save_dir+'mean_peak_swe_error', dpi=300)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# In[ ]:


x = pswe_mean_diff_df.index.values
y_bot = [prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, 0, 0]
y_dif = [pswe_mean_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_mean_peak_swe_bias,0]

lbls = [f'{round(item, 2)}%' for item in y_dif]

# creating error
lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
lb_diff = pswe_mean_diff_df - lb
ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
ub_diff = ub - pswe_mean_diff_df
xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

# plt.figure(figsize=(8,5))
plt.figure(figsize=(16,8))
# bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
h = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.85, 1]
hnew = [i for i in h]
bars = plt.barh(x, y_dif, xerr=xerror, height=hnew, left=y_bot, ecolor='k', error_kw=dict(lw=2, capsize=4, capthick=2))
# plt.bar_label(bars, labels=lbls)    # label bars
plt.title('Mean Peak SWE Error', fontsize=24)
plt.xlabel('Peak SWE Difference (%)', fontsize=20)
# plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--', linewidth=2.5)
plt.axvline(prism_mean_peak_swe_bias, color='grey', alpha=1.0, linestyle='--', linewidth=2.5)
for color, bar in zip(['blue','deepskyblue','deepskyblue','palevioletred','skyblue','skyblue','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-50,50)

# Working with axes
for d in ["left", "top", "right"]:   # set plot spines to invisible
    plt.gca().spines[d].set_visible(False)
plt.tick_params(axis='y', left=False)   # remove tick marks, keep tick labels

# Text size/fonts
# f = [16, 16, 16, 16, 16, 16, 16, 18, 20]
# fnew = [i for i in f]
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# import pylab as plt
# plt.rc('text', usetex=True)
# plt.ylabel(r'\small{Text 1} \Huge{Text2}')

plt.tick_params(axis="y",direction="in", pad=-25)
    
# plt.savefig('/global/cscratch1/sd/yum/swe/conferences/MtnClim/'+'mean_peak_swe_error', dpi=300)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# ### Barplot of Median Peak SWE

# In[ ]:


pswe_med_diff_list = list(pswe_diff_df.median())


# In[ ]:


pswe_med_diff_df = pd.DataFrame(pswe_med_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


prism_med_peak_swe_bias = pswe_med_diff_df['peak_swe_diff']['PRISM']


# In[ ]:


x = pswe_med_diff_df.index.values
y_bot = [prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, 0, 0]
y_dif = [pswe_med_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_med_peak_swe_bias,0]

lbls = [f'{round(item, 2)}%' for item in y_dif]

# creating error
lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
lb_diff = pswe_med_diff_df - lb
ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
ub_diff = ub - pswe_med_diff_df
xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

plt.figure(figsize=(8,5))
bars = plt.barh(x, y_dif, xerr=xerror, height=0.8, left=y_bot, ecolor='k', error_kw=dict(lw=1.25, capsize=2.5, capthick=1.25))
# plt.bar_label(bars, fmt='%.2f')
plt.bar_label(bars, labels=lbls)
plt.title('Median peak SWE error')
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(prism_med_peak_swe_bias, color='grey', alpha=0.75, linestyle='--')
for color, bar in zip(['red','deepskyblue','deepskyblue','palevioletred','skyblue','lightsalmon','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-75,75)
# plt.savefig(save_dir+'median_peak_swe_error', dpi=300)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# In[ ]:


x = pswe_med_diff_df.index.values
y_bot = [prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, 0, 0]
y_dif = [pswe_med_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_med_peak_swe_bias,0]

lbls = [f'{round(item, 2)}%' for item in y_dif]

# creating error
lb = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(lb_q)})
lb_diff = pswe_med_diff_df - lb
ub = pd.DataFrame(data={'peak_swe_diff': pswe_diff_df.quantile(ub_q)})
ub_diff = ub - pswe_med_diff_df
xerror = np.vstack((lb_diff['peak_swe_diff'].to_numpy(), ub_diff['peak_swe_diff'].to_numpy()))

# plt.figure(figsize=(8,5))
plt.figure(figsize=(16,8))
# bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
h = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.85, 1]
hnew = [i for i in h]
bars = plt.barh(x, y_dif, xerr=xerror, height=hnew, left=y_bot, ecolor='k', error_kw=dict(lw=2, capsize=4, capthick=2))
# plt.bar_label(bars, labels=lbls)    # label bars
plt.title('Median Peak SWE Error', fontsize=24)
plt.xlabel('Median SWE Difference (%)', fontsize=20)
# plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--', linewidth=2.5)
plt.axvline(prism_med_peak_swe_bias, color='grey', alpha=1.0, linestyle='--', linewidth=2.5)
for color, bar in zip(['blue','deepskyblue','deepskyblue','palevioletred','skyblue','skyblue','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-50,50)

# Working with axes
for d in ["left", "top", "right"]:   # set plot spines to invisible
    plt.gca().spines[d].set_visible(False)
plt.tick_params(axis='y', left=False)   # remove tick marks, keep tick labels

# Text size/fonts
# f = [16, 16, 16, 16, 16, 16, 16, 18, 20]
# fnew = [i for i in f]
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# import pylab as plt
# plt.rc('text', usetex=True)
# plt.ylabel(r'\small{Text 1} \Huge{Text2}')

plt.tick_params(axis="y",direction="in", pad=-25)
    
# plt.savefig('/global/cscratch1/sd/yum/swe/conferences/MtnClim/'+'med_peak_swe_error', dpi=300)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# In[ ]:





# # Calculate bias of each peak SWE - 08/19/22

# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       calculate difference in peak SWE (compared to SNOTEL baseline)
# boxplot of peak SWE differences 


# ## Compute peak SWE (for each station and year)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


# initialize lists
site_code = []
year = []
pswe_date_snotel, pswe_snotel = [], []
pswe_date_ua, pswe_ua = [], [] 
pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)
            


# ## Compute Biases/Differences

# In[ ]:


# UA difference
ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel)) - 1


# ### Level 1

# In[ ]:


snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel)) - 1


# ### Level 2

# In[ ]:


prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1


# ### Level 3

# #### 3ai

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


pgridmet_bias = ((np.array(pswe_pgridmet_tprism) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


plivneh_bias = ((np.array(pswe_plivneh_tprism) / np.array(pswe_pprism_tprism)) - 1)


# #### 3aii

# ##### Gridmet

# In[ ]:


# (pswe_pprism_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


tgridmet_bias = ((np.array(pswe_pprism_tgridmet) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_pprism_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


tlivneh_bias = ((np.array(pswe_pprism_tlivneh) / np.array(pswe_pprism_tprism)) - 1)


# #### 3aiii

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


ptgridmet_bias = ((np.array(pswe_pgridmet_tgridmet) / np.array(pswe_pprism_tprism)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


ptlivneh_bias = ((np.array(pswe_plivneh_tlivneh) / np.array(pswe_pprism_tprism)) - 1)


# #### 3b

# In[ ]:


# (pswe_pprism_tprism_thjennings_smean / pswe_snotel_smean) - 1


# In[ ]:


thjennings_bias = ((np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_pprism_tprism)) - 1)


# In[ ]:





# ## Plots

# In[ ]:


pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])


# In[ ]:


# convert to %
pswe_diff_df = pswe_diff_df*100


# In[ ]:


# flip order of runs (i.e. columns)
pswe_diff_df = pswe_diff_df.iloc[:, ::-1]


# ### Boxplot of all peak SWE differences

# In[ ]:


# colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

fig,ax = plt.subplots(figsize=(10,6)) 
plt.axvline(0, color='black')
ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)
plt.grid()
plt.xlim(-100,100)
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.title('Peak SWE Bias')
# plt.savefig(save_dir+'boxplot_peak_swe_bias', dpi=300)
# plt.show()


# ### Barplot of Mean Peak SWE

# In[ ]:


pswe_mean_diff_list = list(pswe_diff_df.mean())


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y_bot = np.linspace(30, 50, 10)
y_dif = np.linspace(10, 5, 10)

plt.barh(x, y_dif, left=y_bot)


# In[ ]:


prism_mean_peak_swe_bias = pswe_mean_diff_df['peak_swe_diff']['PRISM']


# In[ ]:


prism_mean_peak_swe_bias


# In[ ]:


x = pswe_mean_diff_df.index.values
y_bot = [prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, prism_mean_peak_swe_bias, 0, 0]
y_dif = [pswe_mean_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_mean_peak_swe_bias,0]   

lbls = [f'{round(item, 2)}%' for item in y_dif]

plt.figure(figsize=(8,5))
bars = plt.barh(x, width=y_dif, height=0.6, left=y_bot)
# plt.bar_label(bars, fmt='%.2f')
plt.bar_label(bars, labels=lbls)
plt.title('Peak SWE error')
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(prism_mean_peak_swe_bias, color='grey', alpha=0.75, linestyle='--')
for color, bar in zip(['red','deepskyblue','deepskyblue','palevioletred','skyblue','lightsalmon','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-100,100)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# ### Barplot of Median Peak SWE

# In[ ]:


pswe_med_diff_list = list(pswe_diff_df.median())


# In[ ]:


pswe_med_diff_df = pd.DataFrame(pswe_med_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


prism_med_peak_swe_bias = pswe_med_diff_df['peak_swe_diff']['PRISM']


# In[ ]:


x = pswe_med_diff_df.index.values
y_bot = [prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, prism_med_peak_swe_bias, 0, 0]
y_dif = [pswe_med_diff_df['peak_swe_diff'][i] for i in range(7)] + [prism_med_peak_swe_bias,0]

lbls = [f'{round(item, 2)}%' for item in y_dif]

plt.figure(figsize=(8,5))
bars = plt.barh(x, y_dif, height=0.6, left=y_bot)
# plt.bar_label(bars, fmt='%.2f')
plt.bar_label(bars, labels=lbls)
plt.title('Peak SWE error')
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(prism_med_peak_swe_bias, color='grey', alpha=0.75, linestyle='--')
for color, bar in zip(['red','deepskyblue','deepskyblue','palevioletred','skyblue','lightsalmon','lightsalmon','darkred','blue'], bars.patches):
    bar.set_color(color)
plt.xlim(-100,100)
# https://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh


# In[ ]:





# # Calculate bias of each peak SWE and then average biases - 07/29/22 Brainstorm #2

# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       calculate difference in peak SWE (compared to SNOTEL baseline)
# boxplot of peak SWE differences 


# ## Compute peak SWE (for each station and year)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


# initialize lists
site_code = []
year = []
pswe_date_snotel, pswe_snotel = [], []
pswe_date_ua, pswe_ua = [], [] 
pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)
            


# ## Compute Biases/Differences

# In[ ]:


# UA difference
ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel)) - 1


# ### Level 1

# In[ ]:


snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel)) - 1


# ### Level 2

# In[ ]:


prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1


# ### Level 3

# #### 3ai

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


pgridmet_bias = ((np.array(pswe_pgridmet_tprism) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


plivneh_bias = ((np.array(pswe_plivneh_tprism) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# #### 3aii

# ##### Gridmet

# In[ ]:


# (pswe_pprism_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


tgridmet_bias = ((np.array(pswe_pprism_tgridmet) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_pprism_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


tlivneh_bias = ((np.array(pswe_pprism_tlivneh) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# #### 3aiii

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


ptgridmet_bias = ((np.array(pswe_pgridmet_tgridmet) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


ptlivneh_bias = ((np.array(pswe_plivneh_tlivneh) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# #### 3b

# In[ ]:


# (pswe_pprism_tprism_thjennings_smean / pswe_snotel_smean) - 1


# In[ ]:


thjennings_bias = ((np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_pprism_tprism)) - 1) + ((np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1)


# In[ ]:





# ## Plots

# ### Boxplot of all peak SWE differences

# In[ ]:


pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])


# In[ ]:


# convert to %
pswe_diff_df = pswe_diff_df*100


# In[ ]:


# flip order of runs (i.e. columns)
pswe_diff_df = pswe_diff_df.iloc[:, ::-1]


# In[ ]:


# colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

fig,ax = plt.subplots(figsize=(10,6)) 
plt.axvline(0, color='black')
ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)
plt.grid()
plt.xlim(-100,100)
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.title('Peak SWE Bias')
plt.savefig(save_dir+'boxplot_peak_swe_bias', dpi=300)
# plt.show()


# ### Barplot of Mean Peak SWE

# In[ ]:


pswe_mean_diff_list = list(pswe_diff_df.mean())


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


ax = pswe_mean_diff_df.plot(kind='barh', figsize=(8,5))
ax.bar_label(ax.containers[0], fmt='%.2f')
ax.set_xlim(-100,100)
ax.axvline(0, color='black')
ax.set_title('Peak SWE bias')
ax.set_xlabel('Peak SWE difference (%)')
ax.set_ylabel('Run')
for color, bar in zip(['red','palevioletred','palevioletred','palevioletred','lightsalmon','lightsalmon','lightsalmon','darkred','blue'], ax.patches):
    bar.set_color(color)
plt.savefig(save_dir+'barplot_mean_peak_swe_bias.png', dpi=300)


# In[ ]:





# # Calculate bias of each peak SWE and then average biases - 07/29/22 Brainstorm #1

# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       calculate difference in peak SWE (compared to SNOTEL baseline)
# boxplot of peak SWE differences 


# ## Compute peak SWE (for each station and year)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


# initialize lists
site_code = []
year = []
pswe_date_snotel, pswe_snotel = [], []
pswe_date_ua, pswe_ua = [], [] 
pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)
            


# ## Compute Biases/Differences

# In[ ]:


# UA difference
ua_bias = (np.array(pswe_ua) / np.array(pswe_snotel)) - 1


# ### Layer 1

# In[ ]:


snotel_bias = (np.array(pswe_snotel) / np.array(pswe_snotel)) - 1


# ### Layer 2

# In[ ]:


prism_bias = (np.array(pswe_pprism_tprism) / np.array(pswe_snotel)) - 1


# ### Layer 3

# #### 3ai

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


pgridmet_bias = ((np.array(pswe_pgridmet_tprism) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


plivneh_bias = ((np.array(pswe_plivneh_tprism) / np.array(pswe_snotel)) - 1)


# #### 3aii

# ##### Gridmet

# In[ ]:


# (pswe_pprism_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


tgridmet_bias = ((np.array(pswe_pprism_tgridmet) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_pprism_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


tlivneh_bias = ((np.array(pswe_pprism_tlivneh) / np.array(pswe_snotel)) - 1)


# #### 3aiii

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


ptgridmet_bias = ((np.array(pswe_pgridmet_tgridmet) / np.array(pswe_snotel)) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


ptlivneh_bias = ((np.array(pswe_plivneh_tlivneh) / np.array(pswe_snotel)) - 1)


# #### 3b

# In[ ]:


# (pswe_pprism_tprism_thjennings_smean / pswe_snotel_smean) - 1


# In[ ]:


thjennings_bias = ((np.array(pswe_pprism_tprism_thjennings) / np.array(pswe_snotel)) - 1)


# In[ ]:





# ## Plots

# ### Boxplot of all peak SWE differences

# In[ ]:


pswe_diff_df = pd.DataFrame(list(zip(snotel_bias, prism_bias, pgridmet_bias, tgridmet_bias, ptgridmet_bias, plivneh_bias, tlivneh_bias, ptlivneh_bias, thjennings_bias)), 
                                 columns = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'])


# In[ ]:


# convert to %
pswe_diff_df = pswe_diff_df*100


# In[ ]:


# flip order of runs (i.e. columns)
pswe_diff_df = pswe_diff_df.iloc[:, ::-1]


# In[ ]:


# colors = ['w', 'w', 'c', 'y', 'm', 'c', 'y', 'm']
colors = ['b', 'm', 'y', 'c', 'm', 'y', 'c', 'w', 'w']

fig,ax = plt.subplots(figsize=(10,6)) 
plt.axvline(0, color='black')
ax,props = pswe_diff_df.plot.box(patch_artist=True, return_type='both', ax=ax, vert=False)
for patch,color in zip(props['boxes'],colors):
    patch.set_facecolor(color)
plt.grid()
plt.xlim(-100,100)
plt.xlabel('Peak SWE difference (%)')
plt.ylabel('Run')
plt.title('Peak SWE Bias')
# plt.savefig(save_dir+'boxplot_peak_swe', dpi=300)
# plt.show()


# ### Barplot of Mean Peak SWE

# In[ ]:


pswe_mean_diff_list = list(pswe_diff_df.mean())


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['JenningsTh', 'LivnehPT', 'LivnehT', 'LivnehP', 'GridMetPT', 'GridMetT', 'GridMetP', 'PRISM', 'SNOTEL'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


ax = pswe_mean_diff_df.plot(kind='barh', figsize=(8,5))
ax.bar_label(ax.containers[0], fmt='%.2f')
ax.set_xlim(-100,100)
ax.axvline(0, color='black')
ax.set_title('Peak SWE bias')
ax.set_xlabel('Peak SWE difference (%)')
ax.set_ylabel('Run')
for color, bar in zip(['red','palevioletred','palevioletred','palevioletred','lightsalmon','lightsalmon','lightsalmon','darkred','blue'], ax.patches):
    bar.set_color(color)
# plt.savefig(save_dir+'peak_swe_bias.png', dpi=300)


# In[ ]:





# # Average peak SWE then calculate bias

# In[ ]:


# for each experiment:
#    for each station:
#       get peak SWE (1 april SWE) across years (for each year)
#       average peak SWE across years
#    average peak SWE across stations
#    calculate difference (compared to SNOTEL baseline)


# ## Compute peak SWE means (across years and stations)

# In[ ]:


def get_peak_swe(df, column):    # adapted from swe_triange_metrics function in swe_triangle_metrics.ipynb
    if column == 'snotel_swe':
        swe = df['snotel_swe']
    elif column == 'ua_swe':
        swe = df['ua_swe']
    elif column == 'my_scaled_swe':
        swe = df['final_scaled_swe']
    elif column == 'my_krig_scaled_swe':
        swe = df['krig_scaled_swe']
    else:
        raise ValueError('Unexpected column.')    
    
    # SPD (peak swe) date
    spd_date = pd.to_datetime(df['datetime'].loc[swe.idxmax()])   #spd_date_yr = pd.to_datetime(df['datetime'].loc[df['final_scaled_swe'].idxmax()])
    # SPD SWE depth
    spd_depth = swe.loc[swe.idxmax()]   #spd_swe_yr = df['final_scaled_swe'].loc[df['final_scaled_swe'].idxmax()]
    
    return spd_date, spd_depth


# In[ ]:


def peak_swe_arr(df, column, year, date_run, metric_run):    # adapted from metrics_arr function in swe_triangle_metrics.ipynb
    yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,7,31)) & (pd.to_datetime(df['datetime']) < datetime(year+1,8,1))]
    # yr_df = df[(pd.to_datetime(df['datetime']) > datetime(year,10,1)) & (pd.to_datetime(df['datetime']) < datetime(year+1,6,30))]
    peak_swe_date, peak_swe_depth = get_peak_swe(yr_df, column)
    # x_run.extend((0, peak_swe_date))
    # y_run.extend((0, peak_swe_depth))
    date_run.append(peak_swe_date)
    metric_run.append(peak_swe_depth)
    return date_run, metric_run


# In[ ]:


pswe_snotel_ymeans = []
pswe_ua_ymeans = []
pswe_pprism_tprism_ymeans = []
pswe_pgridmet_tprism_ymeans = []
pswe_pprism_tgridmet_ymeans = []
pswe_pgridmet_tgridmet_ymeans = []
pswe_plivneh_tprism_ymeans = []
pswe_pprism_tlivneh_ymeans = []
pswe_plivneh_tlivneh_ymeans = []
pswe_pprism_tprism_thjennings_ymeans = []


# In[ ]:


# site_code = []
# year = []
# pswe_date_snotel, pswe_snotel = [], []
# pswe_date_ua, pswe_ua = [], [] 
# pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
# pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
# pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
# pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
# pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
# pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
# pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
# pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []


# In[ ]:


for idx, row in gm_snotel_sites.iterrows():
    sitecode = row['code']
    print(sitecode)
    # site_lon, site_lat = row['geometry'].x, row['geometry'].y
    
    # (re-)initialize lists
    site_code = []
    year = []
    pswe_date_snotel, pswe_snotel = [], []
    pswe_date_ua, pswe_ua = [], [] 
    pswe_date_pprism_tprism, pswe_pprism_tprism = [], []
    pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = [], []
    pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = [], []
    pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = [], []
    pswe_date_plivneh_tprism, pswe_plivneh_tprism = [], []
    pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = [], []
    pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = [], []
    pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = [], []
    
    if os.path.exists(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv') and sitecode != '396_UT_SNTL' and sitecode != '435_UT_SNTL':
        # Load data
        pr_prism_temp_prism_df = pd.read_csv(f'{pr_prism_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_prism_df = pd.read_csv(f'{pr_gridmet_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_gridmet_df = pd.read_csv(f'{pr_prism_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_gridmet_temp_gridmet_df = pd.read_csv(f'{pr_gridmet_temp_gridmet_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_prism_df = pd.read_csv(f'{pr_livneh_temp_prism_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_livneh_df = pd.read_csv(f'{pr_prism_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_livneh_temp_livneh_df = pd.read_csv(f'{pr_livneh_temp_livneh_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        pr_prism_temp_prism_th_jennings_df = pd.read_csv(f'{th_jennings_dir}{sitecode}_concise.csv', index_col='Unnamed: 0')
        
        for yr in np.unique(pd.to_datetime(pr_prism_temp_prism_df['datetime']).dt.year)[:-1]:
            # print(yr)
            site_code.append(sitecode)
            year.append(yr)
            pswe_date_snotel, pswe_snotel = peak_swe_arr(pr_prism_temp_prism_df, 'snotel_swe', yr, pswe_date_snotel, pswe_snotel)
            pswe_date_ua, pswe_ua = peak_swe_arr(pr_prism_temp_prism_df, 'ua_swe', yr, pswe_date_ua, pswe_ua)
            pswe_date_pprism_tprism, pswe_pprism_tprism = peak_swe_arr(pr_prism_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism, pswe_pprism_tprism)
            pswe_date_pgridmet_tprism, pswe_pgridmet_tprism = peak_swe_arr(pr_gridmet_temp_prism_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tprism, pswe_pgridmet_tprism)
            pswe_date_pprism_tgridmet, pswe_pprism_tgridmet = peak_swe_arr(pr_prism_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pprism_tgridmet, pswe_pprism_tgridmet)
            pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet = peak_swe_arr(pr_gridmet_temp_gridmet_df, 'my_scaled_swe', yr, pswe_date_pgridmet_tgridmet, pswe_pgridmet_tgridmet)
            pswe_date_plivneh_tprism, pswe_plivneh_tprism = peak_swe_arr(pr_livneh_temp_prism_df, 'my_scaled_swe', yr, pswe_date_plivneh_tprism, pswe_plivneh_tprism)
            pswe_date_pprism_tlivneh, pswe_pprism_tlivneh = peak_swe_arr(pr_prism_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_pprism_tlivneh, pswe_pprism_tlivneh)
            pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh = peak_swe_arr(pr_livneh_temp_livneh_df, 'my_scaled_swe', yr, pswe_date_plivneh_tlivneh, pswe_plivneh_tlivneh)
            pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings = peak_swe_arr(pr_prism_temp_prism_th_jennings_df, 'my_scaled_swe', yr, pswe_date_pprism_tprism_thjennings, pswe_pprism_tprism_thjennings)

        # compute mean across years (for each station)
        # print(site_code, pswe_snotel)
        pswe_snotel_ymeans.append(statistics.mean(pswe_snotel))
        pswe_ua_ymeans.append(statistics.mean(pswe_ua))
        pswe_pprism_tprism_ymeans.append(statistics.mean(pswe_pprism_tprism))
        pswe_pgridmet_tprism_ymeans.append(statistics.mean(pswe_pgridmet_tprism))
        pswe_pprism_tgridmet_ymeans.append(statistics.mean(pswe_pprism_tgridmet))
        pswe_pgridmet_tgridmet_ymeans.append(statistics.mean(pswe_pgridmet_tgridmet))
        pswe_plivneh_tprism_ymeans.append(statistics.mean(pswe_plivneh_tprism))
        pswe_pprism_tlivneh_ymeans.append(statistics.mean(pswe_pprism_tlivneh))
        pswe_plivneh_tlivneh_ymeans.append(statistics.mean(pswe_plivneh_tlivneh))
        pswe_pprism_tprism_thjennings_ymeans.append(statistics.mean(pswe_pprism_tprism_thjennings))

# average across stations
pswe_snotel_smean = statistics.mean(pswe_snotel_ymeans)
pswe_ua_smean = statistics.mean(pswe_ua_ymeans)
pswe_pprism_tprism_smean = statistics.mean(pswe_pprism_tprism_ymeans)
pswe_pgridmet_tprism_smean = statistics.mean(pswe_pgridmet_tprism_ymeans)
pswe_pprism_tgridmet_smean = statistics.mean(pswe_pprism_tgridmet_ymeans)
pswe_pgridmet_tgridmet_smean = statistics.mean(pswe_pgridmet_tgridmet_ymeans)
pswe_plivneh_tprism_smean = statistics.mean(pswe_plivneh_tprism_ymeans)
pswe_pprism_tlivneh_smean = statistics.mean(pswe_pprism_tlivneh_ymeans)
pswe_plivneh_tlivneh_smean = statistics.mean(pswe_plivneh_tlivneh_ymeans)
pswe_pprism_tprism_thjennings_smean = statistics.mean(pswe_pprism_tprism_thjennings_ymeans)
    
# break


# In[ ]:


pswe_snotel_smean


# In[ ]:


print(pswe_snotel_smean, pswe_ua_smean, pswe_pprism_tprism_smean, pswe_pgridmet_tprism_smean, pswe_pprism_tgridmet_smean,
      pswe_pgridmet_tgridmet_smean, pswe_plivneh_tprism_smean, pswe_pprism_tlivneh_smean, pswe_plivneh_tlivneh_smean, 
      pswe_pprism_tprism_thjennings_smean)


# In[ ]:





# ## Compute Biases/Differences

# In[ ]:


# UA difference
(pswe_ua_smean / pswe_snotel_smean) - 1


# ### Layer 1

# In[ ]:


snotel_bias = (pswe_snotel_smean / pswe_snotel_smean) - 1


# ### Layer 2

# In[ ]:


prism_bias = (pswe_pprism_tprism_smean / pswe_snotel_smean) - 1


# ### Layer 3

# #### 3ai

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


pgridmet_bias = ((pswe_pgridmet_tprism_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tprism_smean / pswe_snotel_smean) - 1


# In[ ]:


plivneh_bias = ((pswe_plivneh_tprism_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# #### 3aii

# ##### Gridmet

# In[ ]:


# (pswe_pprism_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


tgridmet_bias = ((pswe_pprism_tgridmet_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# ##### Livneh

# In[ ]:


# (pswe_pprism_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


tlivneh_bias = ((pswe_pprism_tlivneh_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# #### 3aiii

# ##### Gridmet

# In[ ]:


# (pswe_pgridmet_tgridmet_smean / pswe_snotel_smean) - 1


# In[ ]:


ptgridmet_bias = ((pswe_pgridmet_tgridmet_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# ##### Livneh

# In[ ]:


# (pswe_plivneh_tlivneh_smean / pswe_snotel_smean) - 1


# In[ ]:


ptlivneh_bias = ((pswe_plivneh_tlivneh_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# #### 3b

# In[ ]:


# (pswe_pprism_tprism_thjennings_smean / pswe_snotel_smean) - 1


# In[ ]:


thjennings_bias = ((pswe_pprism_tprism_thjennings_smean / pswe_pprism_tprism_smean) - 1) + ((pswe_pprism_tprism_smean / pswe_snotel_smean) - 1)


# In[ ]:





# ## Barplot

# In[ ]:


pswe_mean_diff_list = [snotel_bias, prism_bias, 
                       pgridmet_bias, tgridmet_bias, ptgridmet_bias, 
                       plivneh_bias, tlivneh_bias, ptlivneh_bias, 
                       thjennings_bias]


# In[ ]:


pswe_mean_diff_df = pd.DataFrame(pswe_mean_diff_list, 
                                 index = ['SNOTEL', 'PRISM', 'GridMetP', 'GridMetT', 'GridMetPT', 'LivnehP', 'LivnehT', 'LivnehPT', 'JenningsTh'],
                                 columns = ['peak_swe_diff'])


# In[ ]:


# convert to %
pswe_mean_diff_df = pswe_mean_diff_df*100


# In[ ]:


# flip order of runs (i.e. rows)
pswe_mean_diff_df = pswe_mean_diff_df.iloc[::-1]


# In[ ]:


ax = pswe_mean_diff_df.plot(kind='barh', figsize=(8,5))
ax.bar_label(ax.containers[0], fmt='%.2f')
ax.set_xlim(-100,100)
ax.axvline(0, color='black')
ax.set_title('Peak SWE bias')
ax.set_xlabel('Peak SWE difference (%)')
ax.set_ylabel('Run')
for color, bar in zip(['red','palevioletred','palevioletred','palevioletred','lightsalmon','lightsalmon','lightsalmon','darkred','blue'], ax.patches):
    bar.set_color(color)
plt.savefig(save_dir+'peak_swe_bias.png', dpi=300)


# In[ ]:





# # TMP

# In[ ]:


import xarray as xr


# In[ ]:


an = xr.open_dataset('/global/cfs/cdirs/risser/PRISM_800M/TOPO_nc/PRISM_TOPO.nc')


# In[ ]:


an.Band1


# In[ ]:





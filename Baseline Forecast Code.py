#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ROOT_PATH = f'~/shared_dir/prod/'
fc_dir = ROOT_PATH + 'forecast_input'
bi_dir = ROOT_PATH + 'reporting'
df = pd.read_parquet(fc_dir + "/ts_ml1.parquet.gzip")
df2 = pd.read_parquet(bi_dir + "/bi.parquet.gzip")


# In[ ]:


#Import the required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from src.utils import cleaning_tools, forecasting_utils
from misc import dbhelper


# In[ ]:


def compute_baseline_forecasts_smape():
    df = pd.read_parquet(fc_dir + "/ts_ml1.parquet.gzip")
    df = df[df['ts']>='2020-10-31']
    df1 = pd.DataFrame({'monthly_maint_plan_qty':df.groupby(['id', 'ts'])['maint_plan_qty'].sum(), 'actuals':df.groupby(['id', 'ts'])['y'].sum()}).reset_index()
    df1['model_name'] = 'baseline'
    df1['month'] = df1['ts'].dt.month
    df2 = pd.DataFrame({'monthly_urgent_breakdown_qty':df.query('wo_demand_indicator==2').groupby(['id', 'ts'])['maint_plan_qty'].sum()}).reset_index()
    id_ts = df1[['id', 'ts']]
    id_ts = id_ts.drop_duplicates(subset=['id', 'ts'])
    df3 = pd.merge(id_ts, df2, on=['id', 'ts'], how='left')
    df3['monthly_urgent_breakdown_qty'] = df3['monthly_urgent_breakdown_qty'].fillna(0)
    df3['mean_urgent_breakdown_qty'] = (df3.groupby('id')['monthly_urgent_breakdown_qty'].apply(lambda x: x.shift().expanding().mean()))
    df4 = pd.merge(df1, df3, on=['id', 'ts'], how='left')
    df4.drop(columns=['monthly_urgent_breakdown_qty'], inplace=True)
    df4['mean_urgent_breakdown_qty'] = df4['mean_urgent_breakdown_qty'].fillna(0)
    df4['forecast'] = df4['monthly_maint_plan_qty'] + df4['mean_urgent_breakdown_qty']
    df4['smape'] = 2 * np.abs((df4['actuals'] - df4['forecast']) / (df4['actuals'] + df4['forecast']))
    df4['smape'] = df4['smape'].fillna(0)
    df4.drop(columns=['monthly_maint_plan_qty', 'mean_urgent_breakdown_qty'], inplace=True)
    cleaning_tools.save_parquet(df4, 'baseline_forecasts.parquet.gzip', index=False, save_dir = '/shared_data/ironman/dev_pr/baseline_stemly_ml_reporting')


# In[ ]:


def generate_baseline_reporting_table():
    df1 = pd.read_parquet("/shared_data/ironman/dev_pr/baseline_forecasts.parquet.gzip")
    df2 = pd.read_parquet(bi_dir + "/bi.parquet.gzip")
    id_material_number = df2[['id', 'material_number']]
    id_material_number = id_material_number.drop_duplicates(subset=['id', 'material_number'], keep='last')
    df2.rename(columns={"sape":"smape"}, inplace=True)
    df2['month'] = df2['ts'].dt.month
    df2 = df2[['id', 'ts', 'actuals', 'model_name', 'month', 'forecast', 'smape']]
    reporting_table = pd.concat([df1, df2])
    reporting_table_final = pd.merge(reporting_table, id_material_number, how='left', on='id')
    reporting_table_final['idx'] = reporting_table_final.index
    cleaning_tools.save_parquet(reporting_table_final, 'bi_with_baseline.parquet.gzip', index=False, save_dir = '/shared_data/ironman/dev_pr/baseline_stemly_ml_reporting')


# In[ ]:


compute_baseline_forecasts_smape()


# In[ ]:


generate_baseline_reporting_table()


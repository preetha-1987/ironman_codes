#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import argparse
import re
import joblib
import matplotlib.pyplot as plt
import glob 
from tqdm import tqdm
import multiprocessing
import time
import warnings
warnings.filterwarnings('ignore')
from p_tqdm import p_map
from src.utils import cleaning_tools, ts_utils, forecasting_utils
from src.config import (
    intermediate_dir,
    primary_dir,
    lookup_dir,
)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)

m1 = None
m2 = None

def process_df_function(df):
    
    output_df = df.copy()

    returns = output_df[output_df.movement_type_inventory_management ==m2]
    previous_issues = pd.DataFrame()
    returns.reset_index(inplace=True, drop=True)

    """
    loop over all returns
    """
    
    for i in range(len(returns)):

        return_qty = returns.iloc[i]['quantity']
        current_date = returns.iloc[i]['posting_date_in_the_document']
        previous_date = returns.iloc[i]['posting_date_in_the_document'] - pd.Timedelta(28, unit='d')

        previous_issues = output_df[(output_df.posting_date_in_the_document >= previous_date) & 
                                    (output_df.posting_date_in_the_document <= current_date) &
                                    (output_df.movement_type_inventory_management == m1)]
        previous_issues.reset_index(drop=True, inplace=True)
        """
        Loop over previous issues for each return
        """
        for j in range(len(previous_issues)-1, -1, -1):
            # make quantity zero
            remaining_returns = return_qty - previous_issues.iloc[j]['quantity']
            previous_issues.loc[j, 'quantity'] = previous_issues.iloc[j]['quantity'] - return_qty
            
            if previous_issues.loc[j, 'quantity'] <= 0:
                previous_issues.loc[j, 'quantity'] = 0
            if remaining_returns > 0:
                return_qty = remaining_returns
            elif remaining_returns <= 0:
                break
        
        if len(previous_issues) == 0:
            None
        else:
            output_df = pd.merge(output_df, previous_issues[['posting_date_in_the_document', 'number_of_material_document',
                                        'item_in_material_document', 'quantity']],
            on=['posting_date_in_the_document', 'number_of_material_document',
                                        'item_in_material_document'],
            how='left')

            output_df['quantity_y'] = output_df['quantity_y'].fillna(output_df['quantity_x'])
            output_df.drop(columns=['quantity_x'], inplace=True)
            output_df.rename(columns={'quantity_y': 'quantity'}, inplace=True)
    return output_df

def read_in_data():
    materials_target = pd.read_parquet(f'{primary_dir}/material_target_with_returns.parquet.gzip')
    movement_type_subset = materials_target[materials_target.movement_type_inventory_management.isin([m1, m2])]

    
def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m1", type=int, required=True, help='Get Previous Issues for this Movement Type')
    parser.add_argument("m2", type=int, required=True, help='Get Returns for this Movement Type')
    args = parser.parse_args()
    global m1
    global m2
    m1 = args.m1
    m2 = args.m2
    movement_type_subset = read_in_data()
    start = time.time()
    print("start chunking")
    gb = movement_type_subset.groupby(['material_number'])
    n_items = list(gb.groups.items())
    n_items = dict(n_items)
    list_groups = [gb.get_group(x) for x in tqdm(n_items,  ascii = True)]
    end = time.time()
    print("chunking done in " + str(end - start))
    start = time.time()
    print("start looping")
    NUM_CORES = 28
    with multiprocessing.Pool(NUM_CORES) as pool:
        if m1==201 and m2==202:
            netted_results_cost = pd.concat(p_map(process_df_function, list_groups), axis=0, ignore_index=True)
            cleaning_tools.save_parquet(netted_results_cost, 'netted_results_cost.parquet.gzip', index=False, save_dir = primary_dir)
        elif m1== 261 and m2==262:
            netted_results_orders = pd.concat(p_map(process_df_function, list_groups), axis=0, ignore_index=True)
            cleaning_tools.save_parquet(netted_results_orders, 'netted_results_orders.parquet.gzip', index=False, save_dir = primary_dir)
        else:
            netted_results_project = pd.concat(p_map(process_df_function, list_groups), axis=0, ignore_index=True)
            cleaning_tools.save_parquet(netted_results_project, 'netted_results_project.parquet.gzip', index=False, save_dir = primary_dir)

if __name__ == '__main__':
    Main()

netted_results_orders = pd.read_parquet(f'{primary_dir}/netted_results_orders.parquet.gzip')
netted_results_project = pd.read_parquet(f'{primary_dir}/netted_results_project.parquet.gzip')
netted_results_cost = pd.read_parquet(f'{primary_dir}/netted_results_cost.parquet.gzip')
netted_all = pd.concat([netted_results_orders, netted_results_project, netted_results_cost])
cleaning_tools.save_parquet(netted_all, 'netted_all.parquet.gzip', index=False, save_dir = primary_dir)


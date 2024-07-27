#!/usr/bin/env python
# coding: utf-8
# %%




import pandas as pd
import numpy as np
import subprocess

# %%


import os
from google.oauth2 import service_account
from googleapiclient.discovery import build


# %%


from constants import *


# %%


import tensorflow as tf


# %%


if False:
    full_df = pl.read_csv_batched('leap-atmospheric-physics-ai-climsim/train.csv', batch_size=100)
    df = full_df.next_batches(1)[0].to_pandas()
    features_df = list(df.columns[1:557])
    targets_df = list(df.columns[557:])
    full_df = pl.read_csv_batched('leap-atmospheric-physics-ai-climsim/train.csv', batch_size=100_000)





from google.cloud import storage





bucket_name="dataleap"




storage_client = storage.Client()
blobs = storage_client.list_blobs(bucket_name)



import xarray as xr
import fsspec




import io
import logging







import xarray as xr






cols_in = ['state_t',   'state_q0001', 'state_q0002', 'state_q0003',
         'state_u', 'state_v', 'pbuf_ozone',  'pbuf_CH4', 'pbuf_N2O']
cols_in_static =  ['state_ps', 'pbuf_SOLIN','pbuf_LHFLX','pbuf_SHFLX', 'pbuf_TAUX',  'pbuf_TAUY',
        'pbuf_COSZRS',   'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF',
       'cam_in_ASDIR', 'cam_in_LWUP', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC',
       'cam_in_OCNFRAC','cam_in_SNOWHLAND',]
cols_out_static = [ 'cam_out_NETSW','cam_out_FLWDS', 'cam_out_PRECSC',  'cam_out_PRECC', 'cam_out_SOLS',
       'cam_out_SOLL',     'cam_out_SOLSD','cam_out_SOLLD',]
   
cols_out =  [ 'state_t','state_q0001', 'state_q0002', 'state_q0003', 
       'state_u', 'state_v']




cols_in_array = ['state_t_0',
 'state_t_1',
 'state_t_2',
 'state_t_3',
 'state_t_4',
 'state_t_5',
 'state_t_6',
 'state_t_7',
 'state_t_8',
 'state_t_9',
 'state_t_10',
 'state_t_11',
 'state_t_12',
 'state_t_13',
 'state_t_14',
 'state_t_15',
 'state_t_16',
 'state_t_17',
 'state_t_18',
 'state_t_19',
 'state_t_20',
 'state_t_21',
 'state_t_22',
 'state_t_23',
 'state_t_24',
 'state_t_25',
 'state_t_26',
 'state_t_27',
 'state_t_28',
 'state_t_29',
 'state_t_30',
 'state_t_31',
 'state_t_32',
 'state_t_33',
 'state_t_34',
 'state_t_35',
 'state_t_36',
 'state_t_37',
 'state_t_38',
 'state_t_39',
 'state_t_40',
 'state_t_41',
 'state_t_42',
 'state_t_43',
 'state_t_44',
 'state_t_45',
 'state_t_46',
 'state_t_47',
 'state_t_48',
 'state_t_49',
 'state_t_50',
 'state_t_51',
 'state_t_52',
 'state_t_53',
 'state_t_54',
 'state_t_55',
 'state_t_56',
 'state_t_57',
 'state_t_58',
 'state_t_59',
 'state_q0001_0',
 'state_q0001_1',
 'state_q0001_2',
 'state_q0001_3',
 'state_q0001_4',
 'state_q0001_5',
 'state_q0001_6',
 'state_q0001_7',
 'state_q0001_8',
 'state_q0001_9',
 'state_q0001_10',
 'state_q0001_11',
 'state_q0001_12',
 'state_q0001_13',
 'state_q0001_14',
 'state_q0001_15',
 'state_q0001_16',
 'state_q0001_17',
 'state_q0001_18',
 'state_q0001_19',
 'state_q0001_20',
 'state_q0001_21',
 'state_q0001_22',
 'state_q0001_23',
 'state_q0001_24',
 'state_q0001_25',
 'state_q0001_26',
 'state_q0001_27',
 'state_q0001_28',
 'state_q0001_29',
 'state_q0001_30',
 'state_q0001_31',
 'state_q0001_32',
 'state_q0001_33',
 'state_q0001_34',
 'state_q0001_35',
 'state_q0001_36',
 'state_q0001_37',
 'state_q0001_38',
 'state_q0001_39',
 'state_q0001_40',
 'state_q0001_41',
 'state_q0001_42',
 'state_q0001_43',
 'state_q0001_44',
 'state_q0001_45',
 'state_q0001_46',
 'state_q0001_47',
 'state_q0001_48',
 'state_q0001_49',
 'state_q0001_50',
 'state_q0001_51',
 'state_q0001_52',
 'state_q0001_53',
 'state_q0001_54',
 'state_q0001_55',
 'state_q0001_56',
 'state_q0001_57',
 'state_q0001_58',
 'state_q0001_59',
 'state_q0002_0',
 'state_q0002_1',
 'state_q0002_2',
 'state_q0002_3',
 'state_q0002_4',
 'state_q0002_5',
 'state_q0002_6',
 'state_q0002_7',
 'state_q0002_8',
 'state_q0002_9',
 'state_q0002_10',
 'state_q0002_11',
 'state_q0002_12',
 'state_q0002_13',
 'state_q0002_14',
 'state_q0002_15',
 'state_q0002_16',
 'state_q0002_17',
 'state_q0002_18',
 'state_q0002_19',
 'state_q0002_20',
 'state_q0002_21',
 'state_q0002_22',
 'state_q0002_23',
 'state_q0002_24',
 'state_q0002_25',
 'state_q0002_26',
 'state_q0002_27',
 'state_q0002_28',
 'state_q0002_29',
 'state_q0002_30',
 'state_q0002_31',
 'state_q0002_32',
 'state_q0002_33',
 'state_q0002_34',
 'state_q0002_35',
 'state_q0002_36',
 'state_q0002_37',
 'state_q0002_38',
 'state_q0002_39',
 'state_q0002_40',
 'state_q0002_41',
 'state_q0002_42',
 'state_q0002_43',
 'state_q0002_44',
 'state_q0002_45',
 'state_q0002_46',
 'state_q0002_47',
 'state_q0002_48',
 'state_q0002_49',
 'state_q0002_50',
 'state_q0002_51',
 'state_q0002_52',
 'state_q0002_53',
 'state_q0002_54',
 'state_q0002_55',
 'state_q0002_56',
 'state_q0002_57',
 'state_q0002_58',
 'state_q0002_59',
 'state_q0003_0',
 'state_q0003_1',
 'state_q0003_2',
 'state_q0003_3',
 'state_q0003_4',
 'state_q0003_5',
 'state_q0003_6',
 'state_q0003_7',
 'state_q0003_8',
 'state_q0003_9',
 'state_q0003_10',
 'state_q0003_11',
 'state_q0003_12',
 'state_q0003_13',
 'state_q0003_14',
 'state_q0003_15',
 'state_q0003_16',
 'state_q0003_17',
 'state_q0003_18',
 'state_q0003_19',
 'state_q0003_20',
 'state_q0003_21',
 'state_q0003_22',
 'state_q0003_23',
 'state_q0003_24',
 'state_q0003_25',
 'state_q0003_26',
 'state_q0003_27',
 'state_q0003_28',
 'state_q0003_29',
 'state_q0003_30',
 'state_q0003_31',
 'state_q0003_32',
 'state_q0003_33',
 'state_q0003_34',
 'state_q0003_35',
 'state_q0003_36',
 'state_q0003_37',
 'state_q0003_38',
 'state_q0003_39',
 'state_q0003_40',
 'state_q0003_41',
 'state_q0003_42',
 'state_q0003_43',
 'state_q0003_44',
 'state_q0003_45',
 'state_q0003_46',
 'state_q0003_47',
 'state_q0003_48',
 'state_q0003_49',
 'state_q0003_50',
 'state_q0003_51',
 'state_q0003_52',
 'state_q0003_53',
 'state_q0003_54',
 'state_q0003_55',
 'state_q0003_56',
 'state_q0003_57',
 'state_q0003_58',
 'state_q0003_59',
 'state_u_0',
 'state_u_1',
 'state_u_2',
 'state_u_3',
 'state_u_4',
 'state_u_5',
 'state_u_6',
 'state_u_7',
 'state_u_8',
 'state_u_9',
 'state_u_10',
 'state_u_11',
 'state_u_12',
 'state_u_13',
 'state_u_14',
 'state_u_15',
 'state_u_16',
 'state_u_17',
 'state_u_18',
 'state_u_19',
 'state_u_20',
 'state_u_21',
 'state_u_22',
 'state_u_23',
 'state_u_24',
 'state_u_25',
 'state_u_26',
 'state_u_27',
 'state_u_28',
 'state_u_29',
 'state_u_30',
 'state_u_31',
 'state_u_32',
 'state_u_33',
 'state_u_34',
 'state_u_35',
 'state_u_36',
 'state_u_37',
 'state_u_38',
 'state_u_39',
 'state_u_40',
 'state_u_41',
 'state_u_42',
 'state_u_43',
 'state_u_44',
 'state_u_45',
 'state_u_46',
 'state_u_47',
 'state_u_48',
 'state_u_49',
 'state_u_50',
 'state_u_51',
 'state_u_52',
 'state_u_53',
 'state_u_54',
 'state_u_55',
 'state_u_56',
 'state_u_57',
 'state_u_58',
 'state_u_59',
 'state_v_0',
 'state_v_1',
 'state_v_2',
 'state_v_3',
 'state_v_4',
 'state_v_5',
 'state_v_6',
 'state_v_7',
 'state_v_8',
 'state_v_9',
 'state_v_10',
 'state_v_11',
 'state_v_12',
 'state_v_13',
 'state_v_14',
 'state_v_15',
 'state_v_16',
 'state_v_17',
 'state_v_18',
 'state_v_19',
 'state_v_20',
 'state_v_21',
 'state_v_22',
 'state_v_23',
 'state_v_24',
 'state_v_25',
 'state_v_26',
 'state_v_27',
 'state_v_28',
 'state_v_29',
 'state_v_30',
 'state_v_31',
 'state_v_32',
 'state_v_33',
 'state_v_34',
 'state_v_35',
 'state_v_36',
 'state_v_37',
 'state_v_38',
 'state_v_39',
 'state_v_40',
 'state_v_41',
 'state_v_42',
 'state_v_43',
 'state_v_44',
 'state_v_45',
 'state_v_46',
 'state_v_47',
 'state_v_48',
 'state_v_49',
 'state_v_50',
 'state_v_51',
 'state_v_52',
 'state_v_53',
 'state_v_54',
 'state_v_55',
 'state_v_56',
 'state_v_57',
 'state_v_58',
 'state_v_59',]+['pbuf_ozone_0',
 'pbuf_ozone_1',
 'pbuf_ozone_2',
 'pbuf_ozone_3',
 'pbuf_ozone_4',
 'pbuf_ozone_5',
 'pbuf_ozone_6',
 'pbuf_ozone_7',
 'pbuf_ozone_8',
 'pbuf_ozone_9',
 'pbuf_ozone_10',
 'pbuf_ozone_11',
 'pbuf_ozone_12',
 'pbuf_ozone_13',
 'pbuf_ozone_14',
 'pbuf_ozone_15',
 'pbuf_ozone_16',
 'pbuf_ozone_17',
 'pbuf_ozone_18',
 'pbuf_ozone_19',
 'pbuf_ozone_20',
 'pbuf_ozone_21',
 'pbuf_ozone_22',
 'pbuf_ozone_23',
 'pbuf_ozone_24',
 'pbuf_ozone_25',
 'pbuf_ozone_26',
 'pbuf_ozone_27',
 'pbuf_ozone_28',
 'pbuf_ozone_29',
 'pbuf_ozone_30',
 'pbuf_ozone_31',
 'pbuf_ozone_32',
 'pbuf_ozone_33',
 'pbuf_ozone_34',
 'pbuf_ozone_35',
 'pbuf_ozone_36',
 'pbuf_ozone_37',
 'pbuf_ozone_38',
 'pbuf_ozone_39',
 'pbuf_ozone_40',
 'pbuf_ozone_41',
 'pbuf_ozone_42',
 'pbuf_ozone_43',
 'pbuf_ozone_44',
 'pbuf_ozone_45',
 'pbuf_ozone_46',
 'pbuf_ozone_47',
 'pbuf_ozone_48',
 'pbuf_ozone_49',
 'pbuf_ozone_50',
 'pbuf_ozone_51',
 'pbuf_ozone_52',
 'pbuf_ozone_53',
 'pbuf_ozone_54',
 'pbuf_ozone_55',
 'pbuf_ozone_56',
 'pbuf_ozone_57',
 'pbuf_ozone_58',
 'pbuf_ozone_59',
 'pbuf_CH4_0',
 'pbuf_CH4_1',
 'pbuf_CH4_2',
 'pbuf_CH4_3',
 'pbuf_CH4_4',
 'pbuf_CH4_5',
 'pbuf_CH4_6',
 'pbuf_CH4_7',
 'pbuf_CH4_8',
 'pbuf_CH4_9',
 'pbuf_CH4_10',
 'pbuf_CH4_11',
 'pbuf_CH4_12',
 'pbuf_CH4_13',
 'pbuf_CH4_14',
 'pbuf_CH4_15',
 'pbuf_CH4_16',
 'pbuf_CH4_17',
 'pbuf_CH4_18',
 'pbuf_CH4_19',
 'pbuf_CH4_20',
 'pbuf_CH4_21',
 'pbuf_CH4_22',
 'pbuf_CH4_23',
 'pbuf_CH4_24',
 'pbuf_CH4_25',
 'pbuf_CH4_26',
 'pbuf_CH4_27',
 'pbuf_CH4_28',
 'pbuf_CH4_29',
 'pbuf_CH4_30',
 'pbuf_CH4_31',
 'pbuf_CH4_32',
 'pbuf_CH4_33',
 'pbuf_CH4_34',
 'pbuf_CH4_35',
 'pbuf_CH4_36',
 'pbuf_CH4_37',
 'pbuf_CH4_38',
 'pbuf_CH4_39',
 'pbuf_CH4_40',
 'pbuf_CH4_41',
 'pbuf_CH4_42',
 'pbuf_CH4_43',
 'pbuf_CH4_44',
 'pbuf_CH4_45',
 'pbuf_CH4_46',
 'pbuf_CH4_47',
 'pbuf_CH4_48',
 'pbuf_CH4_49',
 'pbuf_CH4_50',
 'pbuf_CH4_51',
 'pbuf_CH4_52',
 'pbuf_CH4_53',
 'pbuf_CH4_54',
 'pbuf_CH4_55',
 'pbuf_CH4_56',
 'pbuf_CH4_57',
 'pbuf_CH4_58',
 'pbuf_CH4_59',
 'pbuf_N2O_0',
 'pbuf_N2O_1',
 'pbuf_N2O_2',
 'pbuf_N2O_3',
 'pbuf_N2O_4',
 'pbuf_N2O_5',
 'pbuf_N2O_6',
 'pbuf_N2O_7',
 'pbuf_N2O_8',
 'pbuf_N2O_9',
 'pbuf_N2O_10',
 'pbuf_N2O_11',
 'pbuf_N2O_12',
 'pbuf_N2O_13',
 'pbuf_N2O_14',
 'pbuf_N2O_15',
 'pbuf_N2O_16',
 'pbuf_N2O_17',
 'pbuf_N2O_18',
 'pbuf_N2O_19',
 'pbuf_N2O_20',
 'pbuf_N2O_21',
 'pbuf_N2O_22',
 'pbuf_N2O_23',
 'pbuf_N2O_24',
 'pbuf_N2O_25',
 'pbuf_N2O_26',
 'pbuf_N2O_27',
 'pbuf_N2O_28',
 'pbuf_N2O_29',
 'pbuf_N2O_30',
 'pbuf_N2O_31',
 'pbuf_N2O_32',
 'pbuf_N2O_33',
 'pbuf_N2O_34',
 'pbuf_N2O_35',
 'pbuf_N2O_36',
 'pbuf_N2O_37',
 'pbuf_N2O_38',
 'pbuf_N2O_39',
 'pbuf_N2O_40',
 'pbuf_N2O_41',
 'pbuf_N2O_42',
 'pbuf_N2O_43',
 'pbuf_N2O_44',
 'pbuf_N2O_45',
 'pbuf_N2O_46',
 'pbuf_N2O_47',
 'pbuf_N2O_48',
 'pbuf_N2O_49',
 'pbuf_N2O_50',
 'pbuf_N2O_51',
 'pbuf_N2O_52',
 'pbuf_N2O_53',
 'pbuf_N2O_54',
 'pbuf_N2O_55',
 'pbuf_N2O_56',
 'pbuf_N2O_57',
 'pbuf_N2O_58',
 'pbuf_N2O_59',]


# %%


cols_out_array = [ 'ptend_t_0',
 'ptend_t_1',
 'ptend_t_2',
 'ptend_t_3',
 'ptend_t_4',
 'ptend_t_5',
 'ptend_t_6',
 'ptend_t_7',
 'ptend_t_8',
 'ptend_t_9',
 'ptend_t_10',
 'ptend_t_11',
 'ptend_t_12',
 'ptend_t_13',
 'ptend_t_14',
 'ptend_t_15',
 'ptend_t_16',
 'ptend_t_17',
 'ptend_t_18',
 'ptend_t_19',
 'ptend_t_20',
 'ptend_t_21',
 'ptend_t_22',
 'ptend_t_23',
 'ptend_t_24',
 'ptend_t_25',
 'ptend_t_26',
 'ptend_t_27',
 'ptend_t_28',
 'ptend_t_29',
 'ptend_t_30',
 'ptend_t_31',
 'ptend_t_32',
 'ptend_t_33',
 'ptend_t_34',
 'ptend_t_35',
 'ptend_t_36',
 'ptend_t_37',
 'ptend_t_38',
 'ptend_t_39',
 'ptend_t_40',
 'ptend_t_41',
 'ptend_t_42',
 'ptend_t_43',
 'ptend_t_44',
 'ptend_t_45',
 'ptend_t_46',
 'ptend_t_47',
 'ptend_t_48',
 'ptend_t_49',
 'ptend_t_50',
 'ptend_t_51',
 'ptend_t_52',
 'ptend_t_53',
 'ptend_t_54',
 'ptend_t_55',
 'ptend_t_56',
 'ptend_t_57',
 'ptend_t_58',
 'ptend_t_59',
 'ptend_q0001_0',
 'ptend_q0001_1',
 'ptend_q0001_2',
 'ptend_q0001_3',
 'ptend_q0001_4',
 'ptend_q0001_5',
 'ptend_q0001_6',
 'ptend_q0001_7',
 'ptend_q0001_8',
 'ptend_q0001_9',
 'ptend_q0001_10',
 'ptend_q0001_11',
 'ptend_q0001_12',
 'ptend_q0001_13',
 'ptend_q0001_14',
 'ptend_q0001_15',
 'ptend_q0001_16',
 'ptend_q0001_17',
 'ptend_q0001_18',
 'ptend_q0001_19',
 'ptend_q0001_20',
 'ptend_q0001_21',
 'ptend_q0001_22',
 'ptend_q0001_23',
 'ptend_q0001_24',
 'ptend_q0001_25',
 'ptend_q0001_26',
 'ptend_q0001_27',
 'ptend_q0001_28',
 'ptend_q0001_29',
 'ptend_q0001_30',
 'ptend_q0001_31',
 'ptend_q0001_32',
 'ptend_q0001_33',
 'ptend_q0001_34',
 'ptend_q0001_35',
 'ptend_q0001_36',
 'ptend_q0001_37',
 'ptend_q0001_38',
 'ptend_q0001_39',
 'ptend_q0001_40',
 'ptend_q0001_41',
 'ptend_q0001_42',
 'ptend_q0001_43',
 'ptend_q0001_44',
 'ptend_q0001_45',
 'ptend_q0001_46',
 'ptend_q0001_47',
 'ptend_q0001_48',
 'ptend_q0001_49',
 'ptend_q0001_50',
 'ptend_q0001_51',
 'ptend_q0001_52',
 'ptend_q0001_53',
 'ptend_q0001_54',
 'ptend_q0001_55',
 'ptend_q0001_56',
 'ptend_q0001_57',
 'ptend_q0001_58',
 'ptend_q0001_59',
 'ptend_q0002_0',
 'ptend_q0002_1',
 'ptend_q0002_2',
 'ptend_q0002_3',
 'ptend_q0002_4',
 'ptend_q0002_5',
 'ptend_q0002_6',
 'ptend_q0002_7',
 'ptend_q0002_8',
 'ptend_q0002_9',
 'ptend_q0002_10',
 'ptend_q0002_11',
 'ptend_q0002_12',
 'ptend_q0002_13',
 'ptend_q0002_14',
 'ptend_q0002_15',
 'ptend_q0002_16',
 'ptend_q0002_17',
 'ptend_q0002_18',
 'ptend_q0002_19',
 'ptend_q0002_20',
 'ptend_q0002_21',
 'ptend_q0002_22',
 'ptend_q0002_23',
 'ptend_q0002_24',
 'ptend_q0002_25',
 'ptend_q0002_26',
 'ptend_q0002_27',
 'ptend_q0002_28',
 'ptend_q0002_29',
 'ptend_q0002_30',
 'ptend_q0002_31',
 'ptend_q0002_32',
 'ptend_q0002_33',
 'ptend_q0002_34',
 'ptend_q0002_35',
 'ptend_q0002_36',
 'ptend_q0002_37',
 'ptend_q0002_38',
 'ptend_q0002_39',
 'ptend_q0002_40',
 'ptend_q0002_41',
 'ptend_q0002_42',
 'ptend_q0002_43',
 'ptend_q0002_44',
 'ptend_q0002_45',
 'ptend_q0002_46',
 'ptend_q0002_47',
 'ptend_q0002_48',
 'ptend_q0002_49',
 'ptend_q0002_50',
 'ptend_q0002_51',
 'ptend_q0002_52',
 'ptend_q0002_53',
 'ptend_q0002_54',
 'ptend_q0002_55',
 'ptend_q0002_56',
 'ptend_q0002_57',
 'ptend_q0002_58',
 'ptend_q0002_59',
 'ptend_q0003_0',
 'ptend_q0003_1',
 'ptend_q0003_2',
 'ptend_q0003_3',
 'ptend_q0003_4',
 'ptend_q0003_5',
 'ptend_q0003_6',
 'ptend_q0003_7',
 'ptend_q0003_8',
 'ptend_q0003_9',
 'ptend_q0003_10',
 'ptend_q0003_11',
 'ptend_q0003_12',
 'ptend_q0003_13',
 'ptend_q0003_14',
 'ptend_q0003_15',
 'ptend_q0003_16',
 'ptend_q0003_17',
 'ptend_q0003_18',
 'ptend_q0003_19',
 'ptend_q0003_20',
 'ptend_q0003_21',
 'ptend_q0003_22',
 'ptend_q0003_23',
 'ptend_q0003_24',
 'ptend_q0003_25',
 'ptend_q0003_26',
 'ptend_q0003_27',
 'ptend_q0003_28',
 'ptend_q0003_29',
 'ptend_q0003_30',
 'ptend_q0003_31',
 'ptend_q0003_32',
 'ptend_q0003_33',
 'ptend_q0003_34',
 'ptend_q0003_35',
 'ptend_q0003_36',
 'ptend_q0003_37',
 'ptend_q0003_38',
 'ptend_q0003_39',
 'ptend_q0003_40',
 'ptend_q0003_41',
 'ptend_q0003_42',
 'ptend_q0003_43',
 'ptend_q0003_44',
 'ptend_q0003_45',
 'ptend_q0003_46',
 'ptend_q0003_47',
 'ptend_q0003_48',
 'ptend_q0003_49',
 'ptend_q0003_50',
 'ptend_q0003_51',
 'ptend_q0003_52',
 'ptend_q0003_53',
 'ptend_q0003_54',
 'ptend_q0003_55',
 'ptend_q0003_56',
 'ptend_q0003_57',
 'ptend_q0003_58',
 'ptend_q0003_59',
 'ptend_u_0',
 'ptend_u_1',
 'ptend_u_2',
 'ptend_u_3',
 'ptend_u_4',
 'ptend_u_5',
 'ptend_u_6',
 'ptend_u_7',
 'ptend_u_8',
 'ptend_u_9',
 'ptend_u_10',
 'ptend_u_11',
 'ptend_u_12',
 'ptend_u_13',
 'ptend_u_14',
 'ptend_u_15',
 'ptend_u_16',
 'ptend_u_17',
 'ptend_u_18',
 'ptend_u_19',
 'ptend_u_20',
 'ptend_u_21',
 'ptend_u_22',
 'ptend_u_23',
 'ptend_u_24',
 'ptend_u_25',
 'ptend_u_26',
 'ptend_u_27',
 'ptend_u_28',
 'ptend_u_29',
 'ptend_u_30',
 'ptend_u_31',
 'ptend_u_32',
 'ptend_u_33',
 'ptend_u_34',
 'ptend_u_35',
 'ptend_u_36',
 'ptend_u_37',
 'ptend_u_38',
 'ptend_u_39',
 'ptend_u_40',
 'ptend_u_41',
 'ptend_u_42',
 'ptend_u_43',
 'ptend_u_44',
 'ptend_u_45',
 'ptend_u_46',
 'ptend_u_47',
 'ptend_u_48',
 'ptend_u_49',
 'ptend_u_50',
 'ptend_u_51',
 'ptend_u_52',
 'ptend_u_53',
 'ptend_u_54',
 'ptend_u_55',
 'ptend_u_56',
 'ptend_u_57',
 'ptend_u_58',
 'ptend_u_59',
 'ptend_v_0',
 'ptend_v_1',
 'ptend_v_2',
 'ptend_v_3',
 'ptend_v_4',
 'ptend_v_5',
 'ptend_v_6',
 'ptend_v_7',
 'ptend_v_8',
 'ptend_v_9',
 'ptend_v_10',
 'ptend_v_11',
 'ptend_v_12',
 'ptend_v_13',
 'ptend_v_14',
 'ptend_v_15',
 'ptend_v_16',
 'ptend_v_17',
 'ptend_v_18',
 'ptend_v_19',
 'ptend_v_20',
 'ptend_v_21',
 'ptend_v_22',
 'ptend_v_23',
 'ptend_v_24',
 'ptend_v_25',
 'ptend_v_26',
 'ptend_v_27',
 'ptend_v_28',
 'ptend_v_29',
 'ptend_v_30',
 'ptend_v_31',
 'ptend_v_32',
 'ptend_v_33',
 'ptend_v_34',
 'ptend_v_35',
 'ptend_v_36',
 'ptend_v_37',
 'ptend_v_38',
 'ptend_v_39',
 'ptend_v_40',
 'ptend_v_41',
 'ptend_v_42',
 'ptend_v_43',
 'ptend_v_44',
 'ptend_v_45',
 'ptend_v_46',
 'ptend_v_47',
 'ptend_v_48',
 'ptend_v_49',
 'ptend_v_50',
 'ptend_v_51',
 'ptend_v_52',
 'ptend_v_53',
 'ptend_v_54',
 'ptend_v_55',
 'ptend_v_56',
 'ptend_v_57',
 'ptend_v_58',
 'ptend_v_59',]


cols_out_static=[
 'cam_out_NETSW',
 'cam_out_FLWDS',
 'cam_out_PRECSC',
 'cam_out_PRECC',
 'cam_out_SOLS',
 'cam_out_SOLL',
 'cam_out_SOLSD',
 'cam_out_SOLLD']


# %%


df_limits = pd.read_csv("limits_train_percentiles.csv",index_col=0)
df_limits.sort_values(by="max") .head(50)


# %%


var_columns= list(df_limits[df_limits["max"]>df_limits["min"]].index)
novar_columns= list(df_limits[df_limits["max"]<=df_limits["min"]].index)


# %%


var_columns2= list(df_limits[df_limits["std"]>0].index)
novar_columns2= list(df_limits[df_limits["std"]<=0].index)


# %%


df_limits.loc[list(set(var_columns).difference(set(var_columns2)))]


# %%


len(var_columns),len(novar_columns)


# %%


raices = ["state_t","state_q0001","state_q0002","state_q0003","state_u","state_v",
         "pbuf_ozone","pbuf_CH4","pbuf_N2O"]


# %%


f__ = [e for e in df_limits.index]


# %%


import pandas as pd
import tensorflow as tf



#TFRecords boilerplate
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


import tensorflow as tf
import numpy as np

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))





def serialize_example(features, targets):
    """
    Creates a tf.train.Example message ready to be written to a file.
    
    Args:
        features: A list or numpy array of feature values.
        targets: A list or numpy array of target values.
        
    Returns:
        A serialized tf.train.Example.
    """
    feature_dict = {
        'features': _float_feature(features),
        'targets': _float_feature(targets)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()

# %%


raices_test = ["ptend_t","ptend_q0001","ptend_q0002","ptend_q0003","ptend_u","ptend_v",
              ]


# %%


cols_in ,cols_out


# %%


cols_in_static, cols_out_static




blobs = storage_client.list_blobs(bucket_name)

# Initialize a counter
blob_count = 0

# Iterate over the blobs and count them
for blob in blobs:
    blob_count += 1




print(f"Number of blobs in bucket '{bucket_name}': {blob_count}")


# %%


# List blobs (files) in the specified bucket
blobs = storage_client.list_blobs(bucket_name)


# %%


import tqdm
i = 0
count_=0
data_arr=[]
i=0
for blob in tqdm.tqdm(blobs,total=blob_count):
    i+=1
    
    
    if "mli" in blob.name:
        print(blob.name,i)
      
        with open("a2","wb") as file:
            storage_client.download_blob_to_file(f'gs://dataleap/{blob.name}', file)
            ds = xr.open_dataset(f'a2')
            df_in =ds.to_dataframe()
        with open("b2","wb") as file:
            storage_client.download_blob_to_file(f'gs://dataleap/{blob.name.replace("mli","mlo")}', file)
            ds = xr.open_dataset(f'b2')
            df_out =ds.to_dataframe()

        df_out['state_q0002'] = (df_out['state_q0002'] - df_in['state_q0002'])/1200 # Q tendency [kg/kg/s]
        df_out['state_q0003'] = (df_out['state_q0003'] - df_in['state_q0003'])/1200 # Q tendency [kg/kg/s]
        df_out['state_u'] = (df_out['state_u'] - df_in['state_u'])/1200 # U tendency [m/s/s]
        df_out['state_v'] = (df_out['state_v'] - df_in['state_v'])/1200 # V tendency [m/s/s] 
        df_out['state_t'] = (df_out['state_t'] - df_in['state_t'])/1200 # T tendency [K/s]
        df_out['state_q0001'] = (df_out['state_q0001'] - df_in['state_q0001'])/1200 # Q tendency [kg/kg/s]

        df_in_static = pd.DataFrame(df_in[cols_in_static].values.reshape(-1,60, len(cols_in_static)).mean(axis=1),columns=cols_in_static)
        df_in_array =pd.DataFrame(np.transpose(df_in[cols_in].values.reshape(-1,60,len(cols_in)),(0,2,1)).reshape(-1,60*len(cols_in)),columns=cols_in_array)

        df_out_static = pd.DataFrame(df_out[cols_out_static].values.reshape(-1,60, len(cols_out_static)).mean(axis=1),columns=cols_out_static)
        df_out_array =pd.DataFrame(np.transpose(df_out[cols_out].values.reshape(-1,60,len(cols_out)),(0,2,1)).reshape(-1,60*len(cols_out)),columns=cols_out_array)

        data = pd.concat([df_in_static ,df_in_array,df_out_array,df_out_static],axis=1)
        data=data[f__]











        count_+=len(data)


        data_arr.append(data)
 
        if count_>=200_000:
            print(blob.name)




            data_arr=pd.concat(data_arr,axis=0)
            data_arr[var_columns] = (data_arr[var_columns].values -df_limits[["mean"]].transpose()[var_columns].values)/(df_limits[["std"]].transpose()[var_columns].values)
            data_arr=data_arr.astype(np.float32)
            tf_records_name=f"new_record_{blob.name.split('/')[-1].rsplit('.',1)[0]}.tfrec"
            #Writing the TFRecord
            with tf.io.TFRecordWriter( f'{ tf_records_name}') as writer:
                for k in (range( data_arr.shape[0])):
                    row =  data_arr.iloc[k,:]
                    example = serialize_example(row.values[:556],row.values[556:], ) 
                    writer.write(example) 
            subprocess.run(['gsutil', 'cp', tf_records_name, 'gs://dataleap_new_tensors'], check=True)
            #get_ipython().system('gsutil cp {tf_records_name} gs://dataleap_tensors')
            subprocess.run(['rm', tf_records_name], check=True)
            #get_ipython().system('rm {tf_records_name}')
            data_arr=[]
            count_=0   
        subprocess.run(['rm', "a2"], check=True)
        subprocess.run(['rm', "b2"], check=True)

    

  




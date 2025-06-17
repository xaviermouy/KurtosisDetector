# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:38:49 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.tools import list_files
import ecosound
import matplotlib.pyplot as plt
import os
import sqlite3
import pandas as pd
from ecosound.visualization.grapher_builder import GrapherFactory
## ############################################################################
## Input parameters ###########################################################


annot_dir = r'\\nefscdata\PassiveAcoustics\Stellwagen_Old\STAFF\Xavier\kurtosis_detector\results\NEFSC_SBNMS_201906_SB02'
out_dir=os.path.join(annot_dir,'detections_merged')
audio_file_ext = '.wav'

merge_time_tolerance_sec = 1


## ############################################################################
## ############################################################################
#creates output dir
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)

# import detection results
print('Loading detections')
annot = Annotation()
annot.from_raven(annot_dir,recursive=True)

# set missing data
annot.data.loc[:,'audio_sampling_frequency']=0
annot.data.loc[:,'audio_bit_depth']=0
annot.data.loc[:,'UTC_offset']=0

# merge detections
print('Merging detections')
annot = annot.merge_overlapped(time_tolerance_sec=merge_time_tolerance_sec, inplace=False)
annot.to_netcdf(os.path.join(out_dir,'detections_merged.nc'))
annot.to_raven(out_dir,single_file=False)

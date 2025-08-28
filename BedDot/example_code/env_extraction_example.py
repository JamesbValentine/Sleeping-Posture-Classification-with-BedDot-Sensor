#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:57:19 2023

@author: Yingjian
"""

import numpy as np
from BCG_preprocessing import get_envelope, kurtosis, signal_quality_assessment
from scipy.signal import hilbert, savgol_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

Data_path = '../data/all_ctru_good_data_10s_0.25.npy'

dataset = np.load(Data_path)

fs = 100

vis_ids = np.random.choice(range(dataset.shape[0]), 1, replace=False)
for i in vis_ids:
    x = dataset[i, :-6]
    smoothed_envelope = get_envelope(x = x, Fs = fs, 
                                      low = 2, high = 40, m_wave = 'db12',
                                      denoised_method = 'bandpass',
                                      show = True)
   
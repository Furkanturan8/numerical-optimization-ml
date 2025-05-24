#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:05:21 2025

@author: furkanturan
"""

import numpy as np 
import math
import matplotlib.pyplot as plt

# ----- data -------

ti = np.arange(-4.4,0.4)
yi = [0.6294 * math.exp(0.8116 * t) + np.random.random() * 0.004 for t in ti]

# ----- grafik ------

plt.scatter(ti,yi, color="darkred")
plt.xlabel('ti')
plt.xlabel('yi')
plt.title('Dataset 5', fontstyle='italic')
plt.grid(color='green', linestyle ='--', linewidth = 0.1)
plt.show()
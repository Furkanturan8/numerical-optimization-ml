#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 23:05:41 2025

@author: furkanturan
"""

import numpy as np 
import math
import matplotlib.pyplot as plt

# ----- data -------

ti = np.arange(-20, 20, 0.27)
yi = [math.sin(t)/(t) + np.random.random() * 0.1 for t in ti]

# ----- grafik ------

plt.scatter(ti,yi, color="darkred", s=1)
plt.xlabel('ti')
plt.xlabel('yi')
plt.title('Dataset 6', fontstyle='italic')
plt.grid(color='green', linestyle ='--', linewidth = 0.1)
plt.show()
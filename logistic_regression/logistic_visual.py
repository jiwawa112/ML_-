#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10)

y = 1/(1+np.power(np.e,-x))

plt.title('logistic_regression')
plt.plot(x,y)
plt.show()
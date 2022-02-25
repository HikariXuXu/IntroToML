# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:07:02 2022

@author: MH Xu
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.uniform(-1, 1, 10000)
x2 = np.random.uniform(-1, 1, 10000)
x_test = np.random.uniform(-1, 1, 10000)

a = x1 + x2
b = x1 * x2

a_bar = np.mean(a)
b_bar = np.mean(b)
print('a_bar =', a_bar)
print('b_bar =', b_bar)

e_out = (a*x_test+b-x_test**2)**2
print('e_out =', np.mean(e_out))

bias = (a_bar*x_test+b_bar-x_test**2)**2
print('bias =', np.mean(bias))

var = (a*x_test+b-a_bar*x_test-b_bar)**2
print('var =', np.mean(var))

x = np.linspace(-1,1,200)
plt.plot(x, a_bar*x+b_bar)
plt.plot(x, x**2)
plt.legend(['g_bar(x)', 'f(x)'])
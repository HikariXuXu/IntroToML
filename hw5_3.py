# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:40:57 2022

@author: MH Xu
"""
import matplotlib.pyplot as plt
import numpy as np

x_1 = [1,-1]
y_1 = [1,1]
x_2 = [1,-1]
y_2 = [-1,-1]

plt.figure(1)
plt.scatter(x_1, y_1)
plt.scatter(x_2, y_2)
plt.xlabel('x1')
plt.ylabel('x1x2')
plt.legend(['-1','1'])
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.plot(np.linspace(-2,2,100), np.zeros(100), color='r')
plt.plot(np.linspace(-2,2,100), np.ones(100), color='g', linestyle='--')
plt.plot(np.linspace(-2,2,100), -np.ones(100), color='b', linestyle='--')
plt.show()

x_3 = [1,-1]
y_3 = [1,-1]
x_4 = [1,-1]
y_4 = [-1,1]

plt.figure(2)
plt.scatter(x_3, y_3)
plt.scatter(x_4, y_4)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['-1','1'])
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.plot(np.linspace(-2,2,100), np.zeros(100), color='r')
plt.plot(np.zeros(100), np.linspace(-2,2,100), color='r')
plt.plot(np.linspace(-2,-0.1,100), np.ones(100)/np.linspace(-2,-0.1,100), color='g', linestyle='--')
plt.plot(np.linspace(0.1,2,100), np.ones(100)/np.linspace(0.1,2,100), color='g', linestyle='--')
plt.plot(np.linspace(-2,-0.1,100), -np.ones(100)/np.linspace(-2,-0.1,100), color='b', linestyle='--')
plt.plot(np.linspace(0.1,2,100), -np.ones(100)/np.linspace(0.1,2,100), color='b', linestyle='--')

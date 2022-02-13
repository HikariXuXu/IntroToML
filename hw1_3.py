# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def flipCoin(N, times):
    experiment = np.random.randint(2, size=(N, times))
    nu_1 = sum(experiment[0])/times
    nu_rand = sum(experiment[np.random.randint(N)])/times
    nu_min = min(np.sum(experiment, axis=1))/times
    return nu_1, nu_rand, nu_min

nu1 = np.zeros(100000)
nuRand = np.zeros(100000)
nuMin = np.zeros(100000)

np.random.seed(0)

for i in range(100000):
    nu1[i], nuRand[i], nuMin[i] = flipCoin(1000, 10)

print("Printing histogram...")
plt.hist(nu1, bins=11)
plt.xlabel("nu_1")
plt.ylabel("Count")
plt.show()

plt.hist(nuRand, bins=11)
plt.xlabel("nu_rand")
plt.ylabel("Count")
plt.show()

plt.hist(nuMin, bins=np.linspace(0,1,11))
plt.xlabel("nu_min")
plt.ylabel("Count")
plt.xlim(-0.05, 1.05)
plt.show()

epsilon = np.linspace(0,0.5,11)
pr1 = np.zeros(11)
prRand = np.zeros(11)
prMin = np.zeros(11)
for i in range(11):
    pr1[i] = sum(abs(nu1-0.5)>epsilon[i])/100000
    prRand[i] = sum(abs(nuRand-0.5)>epsilon[i])/100000
    prMin[i] = sum(abs(nuMin-0.5)>epsilon[i])/100000

eps = np.linspace(0,0.5,101)

plt.plot(epsilon, pr1)
plt.plot(eps, 2*np.exp(-2*eps**2*10))
plt.xlabel("eps")
plt.ylabel("Prob.")
plt.legend(["Pr[|nu-mu|>eps]", "Hoeffding Bound"])
plt.show()
plt.plot(epsilon, prRand)
plt.plot(eps, 2*np.exp(-2*eps**2*10))
plt.xlabel("eps")
plt.ylabel("Prob.")
plt.legend(["Pr[|nu-mu|>eps]", "Hoeffding Bound"])
plt.show()
plt.plot(epsilon, prMin)
plt.plot(eps, 2*np.exp(-2*eps**2*10))
plt.xlabel("eps")
plt.ylabel("Prob.")
plt.legend(["Pr[|nu-mu|>eps]", "Hoeffding Bound"])
plt.show()
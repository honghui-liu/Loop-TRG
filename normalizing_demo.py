#  
#  normalizing_demo.py
#  Loop_TRG
#  Demonstrate how to find the scaling constant f and gamma (4 -> 2 transform)
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product

# for temperature in np.arange(1.8,2.8,0.05,dtype=float):

temperature = 3
# assign the initial value of tensor TA, TB (2D Ising model)
ts_T_A0 = np.ones((2,2,2,2),dtype=complex)
ts_T_A0[0,1,0,1] = np.exp(-4/temperature)
ts_T_A0[1,0,1,0] = np.exp(-4/temperature)
ts_T_A0[0,0,0,0] = np.exp(4/temperature)
ts_T_A0[1,1,1,1] = np.exp(4/temperature)
ts_T_B0 = ts_T_A0.copy()

# the scaling is applied to the 4 -> 2 process

# Normalized process
# tensor normalization constant (the initial value of gamma)
gamma_A0 = np.einsum('lulu', ts_T_A0)
gamma_B0 = np.einsum('lulu', ts_T_B0)
# normalized tensor before RG
ts_TN_A0 = ts_T_A0 / gamma_A0
ts_TN_B0 = ts_T_B0 / gamma_B0
# normalized partition function for 4 sites
part_ZN0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_TN_A0,ts_TN_B0,ts_TN_A0,ts_TN_B0)
# entanglement filtering
ts_TN_A0_flt, ts_TN_B0_flt = flt.filter(ts_TN_A0, ts_TN_B0, 1.0E-12)
# loop optimize to find the new tensors
# ts_TN_A0_flt -> fA * ts_TN_A1 = ts_T_A1
# ts_TN_B0_flt -> fB * ts_TN_B1 = ts_T_B1
ts_T_A1, ts_T_B1 = opt.loop_optimize((ts_TN_A0_flt,ts_TN_B0_flt), 16, 1E-6)
# tensor normalization constant
fA = np.einsum('lulu', ts_T_A1)
fB = np.einsum('lulu', ts_T_B1)

# Unnormalized process
# partition function for 4 sites
part_Z0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A0,ts_T_B0,ts_T_A0,ts_T_B0)
# entanglement filtering
ts_T_A0_flt, ts_T_B0_flt = flt.filter(ts_T_A0, ts_T_B0, 1.0E-12)
# loop optimize to find the new tensors
# ts_T_A0_flt -> ts_T_A1
# ts_T_B0_flt -> ts_T_B1
ts_T_A1, ts_T_B1 = opt.loop_optimize((ts_T_A0_flt,ts_T_B0_flt), 16, 1E-6)
# tensor normalization constant
gamma_A1 = np.einsum('lulu', ts_T_A1)
gamma_B1 = np.einsum('lulu', ts_T_B1)
# partition function for 2 sites
part_Z1 = np.einsum('dcba,badc', ts_T_A1,ts_T_B1)

# verify:
# (gamma_A0)^2 fA = (gamma_A1)
# (gamma_B0)^2 fB = (gamma_B1)

print(gamma_A0, gamma_B0)
print(fA, fB)
print(fA * (gamma_A0**2), gamma_A1)
print(fB * (gamma_B0**2), gamma_B1)

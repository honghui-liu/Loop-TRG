#  
#  LN_main.py
#  Loop_TRG
#  Main program for LN-TRG calculation
#
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product
import sys

temperature = float(sys.argv[1])
# temperature = 2.0

# assign the initial value of tensor TA, TB (2D Ising model)
ts_T_A0 = np.ones((2,2,2,2),dtype=complex)
ts_T_A0[0,1,0,1] = np.exp(-4/temperature)
ts_T_A0[1,0,1,0] = np.exp(-4/temperature)
ts_T_A0[0,0,0,0] = np.exp(4/temperature)
ts_T_A0[1,1,1,1] = np.exp(4/temperature)
ts_T_B0 = ts_T_A0.copy()

# tensor normalization constant (the initial value of gamma)
gamma_A0 = np.einsum('lulu', ts_T_A0)
gamma_B0 = np.einsum('lulu', ts_T_B0)
# initial value ln(z_0)
ln_z_A = np.log(gamma_A0)
ln_z_B = np.log(gamma_B0)
# normalized tensor before RG
ts_TN_A0 = ts_T_A0 / gamma_A0
ts_TN_B0 = ts_T_B0 / gamma_B0
# initial area per tensor = 1
area = 1

filename = 'LN_temperature' + ('%.3f' % temperature) + '.txt'
with open(filename, 'w+') as f:

    for i in range(16):
        # entanglement filtering
        # ts_TN_A0, ts_TN_B0 = flt.filter(ts_TN_A0, ts_TN_B0, -1.0)
        # loop optimize to find the new tensors
        # ts_TN_A0 -> fA * ts_TN_A1 = ts_T_A1
        # ts_TN_B0 -> fB * ts_TN_B1 = ts_T_B1
        ts_T_A1, ts_T_B1, loop_error = opt.loop_optimize((ts_TN_A0,ts_TN_B0),16, np.inf,0)
        fA = np.einsum('lulu', ts_T_A1)
        fB = np.einsum('lulu', ts_T_B1)
        f.write(str(fA) + '\t' + str(fB) + '\t' + str(loop_error) + '\n')
        # update area
        area *= 2
        # update ln(z)
        ln_z_A += np.log(fA) / area
        ln_z_B += np.log(fB) / area
        # normalized tensor after RG
        ts_TN_A1 = ts_T_A1 / fA
        ts_TN_B1 = ts_T_B1 / fB
        # update the (normalized) tensor
        ts_TN_A0 = ts_TN_A1.copy()
        ts_TN_B0 = ts_TN_B1.copy()

    free_energy_dens = - temperature * (ln_z_A + ln_z_B)
    f.write(str(temperature) + '\t' + str(free_energy_dens) + '\n')

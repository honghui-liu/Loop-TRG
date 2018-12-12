#  
#  RG_test.py
#  Loop_TRG
#  
#  Copyright (C) 2018 Yue Zhengyuan, Liu Honghui and Zhang Wenqian. All rights reserved.
#  Article reference: Phys. Rev. Lett. 118, 110504 (2017)
#

import numpy as np
import filtering as flt
import optimizing as opt
from itertools import product

temperature = 1

# assign the initial value of tensor TA, TB (2D Ising model)
ts_T_A0 = np.ones((2,2,2,2),dtype=complex)
ts_T_A0[0,1,0,1] = np.exp(-4/temperature)
ts_T_A0[1,0,1,0] = np.exp(-4/temperature)
ts_T_A0[0,0,0,0] = np.exp(4/temperature)
ts_T_A0[1,1,1,1] = np.exp(4/temperature)
ts_T_B0 = ts_T_A0.copy()

# partition function for 2 sites
part_Z0 = np.einsum('dcba,badc', ts_T_A0,ts_T_B0)
print(np.log(part_Z0)/2)
# partition function for 4 sites
part_Z0 = np.einsum('ajkb,cbmj,mdci,kiad', ts_T_A0,ts_T_B0,ts_T_A0,ts_T_B0)
print(np.log(part_Z0)/4)
# partition function for 8 sites
part_Z0 = np.einsum('mnfe,pqhg,heib,fgaj,ijkl,abcd,clmq,kdpn',ts_T_A0,ts_T_A0,ts_T_B0,ts_T_B0,ts_T_A0,ts_T_A0,ts_T_B0,ts_T_B0)
print(np.log(part_Z0)/8)
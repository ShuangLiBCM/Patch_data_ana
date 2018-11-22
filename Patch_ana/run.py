### Use process to run the pre-processing
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdb
from Patch_ana import patch_pip
import seaborn as sns
from scipy import stats
plt.style.use('classic')

import warnings
warnings.filterwarnings("ignore")


data_path = "/Users/Shuang/Dropbox/Andreas' Lab/Server Data/Shuang/Processed_test"
data = pd.read_csv(data_path+'/good data storage.csv')

drop_index = [61, 155, 190]
for i in drop_index:
    data.drop(i, inplace=True)

data_reci = data[(data.Reci == 1)&(data['E-I']==1)]

for i in range(100, 120, 20):
    print(f'processing index {i} to {i+20}...')
    patch_pip.df_ana(data_reci.iloc[i:i+21], data_path+'/processed_EI_reci_all_{index}'.format(index=i), datapath=data_path)

print('Done')
import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

data_dir  = './data'
data_file = 'data.xlsx'


Keys = ['時間(s)', 'spindle(RPM)', 'ch1(um)', 'ch2(um)', 'ch3(um)', 'ch4(um)', 'ch5(um)', 't1(°C)', 't2(°C)', 't3(°C)', 't4(°C)', 't5(°C)', 't6(°C)', 't7(°C)', 't8(°C)', 't9(°C)', 't10(°C)', 't11(°C)', 't12(°C)', 't13(°C)', 't14(°C)', 't15(°C)', 't16(°C)']
X_keys = ['t1(°C)', 't2(°C)', 't3(°C)', 't4(°C)', 't5(°C)', 't6(°C)', 't7(°C)', 't8(°C)', 't9(°C)', 't10(°C)', 't11(°C)', 't12(°C)', 't13(°C)', 't14(°C)', 't15(°C)', 't16(°C)'] 
Y_keys = ['ch1(um)', 'ch2(um)', 'ch3(um)', 'ch4(um)', 'ch5(um)']
Data = pd.read_excel(io = os.path.join(data_dir+'/'+data_file), sheet_name = '工作表1', names = Keys)
D_len, D_id = Data.shape
X_data, Y_data = [],[]
swriter = SummaryWriter('./logs/Data')
for id in range(D_id):
    if Keys[id] in X_keys:
        X_data.append(Data[Keys[id]])
        for t in range(D_len):
            swriter.add_scalars('X', {Keys[id]: Data[Keys[id]].iat[t]}, t)
    if Keys[id] in Y_keys:
        Y_data.append(Data[Keys[id]])
        for t in range(D_len):
            swriter.add_scalars('Y', {Keys[id]: Data[Keys[id]].iat[t]}, t)     

pd.concat(X_data, axis=1).to_csv('./logs/Data/X.csv', index=False)  
pd.concat(Y_data, axis=1).to_csv('./logs/Data/Y.csv', index=False)
# pd.concat(X_data+Y_data, axis=1).to_csv('./logs/Data/XY.csv', index=False)  
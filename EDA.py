import os
import pandas as pd
from tqdm.auto import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
import pickle

@lru_cache
def source_filter(s:str):
    if s.startswith('WELL'): return 'WELL'
    elif s.startswith('SIM'): return 'SIM'
    elif s.startswith('DRAWN'): return 'DRAWN'

def main():
    # Dataset directory
    abs_dir = os.getcwd()
    dataset_dir = abs_dir + '\\data'
    dir_list = os.listdir(dataset_dir) # ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    dir_dict = {}
    for l in dir_list: dir_dict[l] = os.listdir(f'{dataset_dir}/{l}')
    labels=['class', 'P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL','source','instance_id']
    p_labels = ['P-PDG', 'P-TPT', 'P-MON-CKP', 'P-JUS-CKGL']

    print(f'Pressure variables are ')
    for l in p_labels:
        print(f'{l}')

    print(f'\nData loading...')
    if 'df_all.pkl' in os.listdir('cached'):
        with open(os.path.join('./cached',f'df_all.pkl'), 'rb') as f:
            df_all = pickle.load(f)
    else:
        lst_all = []
        instance_id=0
        for key in dir_dict.keys():
            lst = [pd.DataFrame(columns=labels)]
            for l in tqdm(dir_dict[key],desc='Class '+key):
                df_ = pd.read_csv(f"{dataset_dir}/{key}/{l}", engine="pyarrow")
                df_['source'] = source_filter(l)
                df_['instance_id'] = instance_id
                # for c in p_labels:
                #     if (df_[c].max() < ub) & (df_[c].min() >= lb):
                #         normal.append((key, instance_id,c,l))
                #     else:
                #         abnormal.append((key, instance_id,c,l))
                instance_id +=1
                lst.append(df_)
            df = pd.concat(lst, axis=0)
            lst_all.append(df)
            # cache
            with open(os.path.join('./cached',f'df_{key}.pkl'), 'wb') as f:
                pickle.dump(df, f)

        print('\nNow caching all of the dataset...\n')
        df_all = pd.concat(lst_all, axis=0)
        df_all.reset_index(drop=True)
        with open(os.path.join('./cached',f'df_all.pkl'), 'wb') as f:
            pickle.dump(df_all, f)
    print('Complete')

    print(colored('\nConduct EDA in terms of the pressure variables\n',attrs=['bold']))
    p = df_all[['P-PDG', 'P-TPT', 'P-MON-CKP', 'P-JUS-CKGL']]
    print('Description')
    print(p.describe())
    print('\nCount')
    print(p.count())
    print('\nBoxplot of the pressure variables')
    plt.boxplot([p[key][True != np.isnan(p[key])] for key in p.keys()])
    plt.show()
    print("=> It has problems with scales")
    print('\n')
    print('='*20)
    print(colored('Logical Threshold\n',attrs=['bold']))
    print(colored('Step 1) Only have the positive values\n',attrs=['bold']))
    logical = (p >= 0)

    print('Ratio of the positive values')
    print(p[logical].count() / p.count())
    print('\nDescription')
    print(p[logical].describe())
    print('\nBoxplot of the pressure variables')
    plt.boxplot([p[logical][key][True != np.isnan(p[logical][key])] for key in p.keys()])
    plt.show()
    print("=>It still has problems with scales")

    print(colored('\nStep 2) Elimination of the hard-outlier in the positive values\n',attrs=['bold']))
    p_logical = p[logical]
    IQR = p_logical.quantile(0.75) - p.quantile(0.25)
    ub = p_logical.quantile(0.75) + 3 * IQR
    lb = p_logical.quantile(0.25) - 3 * IQR
    lb[lb < 0] = 0
    logical_IQR = (p_logical < ub) & (p_logical >= lb)

    print('Ratio of the logical values')
    print(p_logical[logical_IQR].count()/p.count())
    print('\nDescription')
    print(p_logical[logical_IQR].describe())
    print('\nBoxplot of the pressure variables')
    plt.boxplot([p_logical[logical_IQR][key][True != np.isnan(p_logical[logical_IQR][key])] for key in p.keys()])
    plt.show()
    print(colored("Logical threshold is confirmed!\n", 'red'))

    print("bound")
    bound = pd.concat([lb, ub], axis=1)
    bound.columns = ['lb', 'ub']
    with open('./cached/bound.pkl', 'wb') as f:
        pickle.dump(bound, f)
    print(bound)
if __name__ == "__main__":
    main()




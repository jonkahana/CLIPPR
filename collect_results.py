import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import express as px
import os
from os.path import join
import re

dataset = 'utk'
is_running = True
res_folder = join('weights', dataset)

res_files_alias = 'Results.txt'
files_alias_name = res_files_alias.split('.')[0]

keywords = ['MAE', 'Accuracy', 'Loss']

if __name__ == '__main__':

    results_dict = {}

    for dir in os.listdir(res_folder):

        exp_dir = join(res_folder, dir)
        exp_dict = {'exp_name': dir, 'dataset': dataset}

        if not os.path.isdir(exp_dir):
            continue

        # make sure only one file in each directory matches the alias
        matching_files_in_dir = [x for x in os.listdir(exp_dir) if res_files_alias in x]
        if len(matching_files_in_dir) == 0:
            if not is_running:
                raise ValueError(f'Zero results file matches to the search in the directory: {exp_dir}')
            else:
                continue
        elif len(matching_files_in_dir) != 1:
            raise ValueError(f'More than one results file matches to the search in the directory: {exp_dir}\n{matching_files_in_dir}')
        exp_res_filename = [x for x in os.listdir(exp_dir) if res_files_alias in x][0]
        exp_res_filepath = join(exp_dir, exp_res_filename)
        with open(exp_res_filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            split_line = re.split(':\s+', line)
            if len(split_line) == 2:
                key, value = split_line

                # replace key by keyword
                if 'MAE' in key and 'Loss' in key:
                    key = 'MAE'
                elif np.sum([k in key for k in keywords]) != 1:
                    raise ValueError(f'Illegal number of keywords in \"{key}\"')
                else:
                    key = keywords[np.where([k in key for k in keywords])[0][0]]

                # value = float(value)
                exp_dict[key] = value
            else:
                raise ValueError(f'Value missed in file {exp_res_filepath}')

        results_dict[dir] = exp_dict

    results_df = pd.DataFrame(results_dict).T
    results_df = results_df.sort_values(by='exp_name')
    results_df.to_csv(join(res_folder, f'{files_alias_name}__results_summary.csv'))

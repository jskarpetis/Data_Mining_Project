import csv
from fileinput import filename
from operator import index
from typing import final
import pandas as pd
import numpy as np
import os
import sklearn


def merge_csv(csv_header, folder_to_merge):
    csv_directory = os.getcwd() + f'\dataset\{folder_to_merge}\\'
    csv_output = os.getcwd() + f'/dataset/consolidated_{folder_to_merge}.csv'
    directory_tree = os.walk(csv_directory)
    csv_list = []

    for directory_path, dir_names, file_names in directory_tree:
        for file in file_names:
            if file.endswith('.csv'):
                csv_list.append(file)

    csv_merge = open(csv_output, 'w')
    csv_merge.write(csv_header)
    csv_merge.write('\n')

    for file in csv_list:
        csv_in = open(directory_path + file, 'r')
        csv_in = csv_in.readlines()

        if (folder_to_merge=='demand' and not len(csv_in) == 0):
            csv_in.pop()
        for line in csv_in:
            if line.lower().startswith(csv_header.lower()):
                continue
            
            csv_merge.write(line)
    csv_merge.close()

    print('Verify consolidated CSV file : ' + csv_output)


if __name__ == "__main__":

    merge_csv(csv_header='Time,Day ahead forecast,Hour ahead forecast,Current demand',
              folder_to_merge='demand')
    merge_csv(csv_header='Time,Solar,Wind,Geothermal,Biomass,Biogas,Small hydro,Coal,Nuclear,Natural gas,Large hydro,Batteries,Imports,Other', folder_to_merge='sources')

    demand_merged_dataset = pd.read_csv(
        'dataset/consolidated_demand.csv')
    sources_merged_dataset = pd.read_csv(
        'dataset/consolidated_sources.csv')

    # Axis 1 means its going to put the elements side by side
    final_dataset = pd.concat(
        [demand_merged_dataset, sources_merged_dataset], axis=1)
    final_dataset = final_dataset.fillna(0)

    # Storing the final csv, with merged data
    final_dataset.to_csv(os.getcwd() + '/dataset/final_dataset.csv')

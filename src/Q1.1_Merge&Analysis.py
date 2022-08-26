import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt

def merge_csv(csv_header, folder_to_merge):

    dataset_path = os.getcwd()
    csv_directory = dataset_path + f'\\dataset\\{folder_to_merge}\\'
    csv_output = dataset_path + f'\\dataset\\consolidated_{folder_to_merge}.csv'
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

def plot_data(dF):
    figure, axis = plt.subplots(4,4, figsize=(7,7))
    figure.tight_layout()
    columns = dF.columns.tolist()
    row = 0

    for index, elem in enumerate(columns):

        axis[row, index%4].plot(dF[elem])
        axis[row, index%4].set_title(elem)
        if (index%4 == 3):
            row += 1
    plt.show()

if __name__ == "__main__":

    # merge_csv(csv_header='Time, \
    #                     Day ahead forecast,\
    #                     Hour ahead forecast,\
    #                     Current demand',
    #                     folder_to_merge='demand')

    # merge_csv(csv_header='Time,\
    #                     Solar,\
    #                     Wind,\
    #                     Geothermal,\
    #                     Biomass,\
    #                     Biogas,\
    #                     Small hydro,\
    #                     Coal,\
    #                     Nuclear,\
    #                     Natural gas,\
    #                     Large hydro,\
    #                     Batteries,\
    #                     Imports,\
    #                     Other', folder_to_merge='sources')

    # Merge the two big datasets of demands and sources in a single csv
    # demand_merged_dataset = pd.read_csv(r'\dataset\consolidated_demand.csv')
    # sources_merged_dataset = pd.read_csv(r'\dataset\consolidated_sources.csv')

    # Axis 1 means its going to put the elements side by side
    # final_dataset = pd.concat([demand_merged_dataset, sources_merged_dataset], axis=1)
    # final_dataset = final_dataset.fillna(0)

    # Storing the final csv, with merged data
    # final_dataset.to_csv(os.getcwd() + r'\dataset\final_dataset.csv')

    path = os.getcwd() + r'\dataset\final_dataset.csv'
    final_dataset = pd.read_csv(path)
    
    dF = final_dataset.loc[:,['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    
    print('Final stats -->\n {}'.format(dF.describe()))

    plot_data(dF)

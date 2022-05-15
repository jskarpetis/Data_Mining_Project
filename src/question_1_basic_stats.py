
from cProfile import label
import csv
from fileinput import filename
from operator import index
from typing import final
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import statistics

def produce_basic_stats(dataframe):
    columns = dataframe.columns
    stats = {}
    for column in columns:
        
        # Getting each columns data to calculate stats
        data = dataframe.loc[:,column]
        
        data_min = np.round(min(data), 2)
        data_max = np.round(max(data), 2)
        data_mean = np.round(data.mean(), 2)
        data_std = np.round(data.std(), 2)
        data_var = np.round(data.var(), 2)
        
        stats[column] =  [data_min, data_max, data_mean, data_std, data_var]

    final_stats = pd.DataFrame(stats, index=['data_min', 'data_max', 'data_mean', 'data_std', 'data_var'])
    return final_stats
    
        
        
def plot(dF):
    
    figure, axis = plt.subplots(4,4, figsize=(7,7))
    figure.tight_layout()
    columns = dF.columns.tolist()
    row = 0

    for index, elem in enumerate(columns):

        axis[row, index%4].plot(dF[elem])
        axis[row, index%4].set_title(elem)
        if (index%4 == 3):
            row += 1

    # Showing in full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
        
        
            


if __name__ == "__main__":
    path = os.getcwd() + '/dataset/final_dataset.csv'
    final_dataset = pd.read_csv(path)
    
    dF = final_dataset.loc[:,['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    final_stats = produce_basic_stats(dF)
    print('Final stats -->\n {}'.format(final_stats))
    
    
    # for i in range(0, len(final_dataset), 289):
    #     batch_to_plot = final_dataset[i:i+289]
    #     plot(batch_to_plot)
    plot(dF)
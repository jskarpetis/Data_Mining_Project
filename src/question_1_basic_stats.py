
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

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
    path = os.getcwd() + '/dataset/final_dataset.csv'
    final_dataset = pd.read_csv(path)
    
    dF = final_dataset.loc[:,['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    
    print('Final stats -->\n {}'.format(dF.describe()))

    plot_data(dF)
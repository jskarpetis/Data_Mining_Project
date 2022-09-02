import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if __name__ == "__main__":
    batch_size = 288

    dataframe = pd.read_csv(os.getcwd() + '/dataset/final_dataset.csv')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.loc[:,['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]

    columns = dataframe.columns

    # New split dataset
    day_split_dataset = pd.DataFrame(columns=columns)

    for column in columns:
        counter = 0
        # Getting data of each column
        data = dataframe.loc[:,column]
        # Spliting data per day 
        mini_batches = [data[k: k + batch_size] for k in range(0, len(data), batch_size)]
        for mini_batch in mini_batches:
            # For every batch and column find mean and place it in new dataset
            mean = np.round(mini_batch.mean(), 2)
            day_split_dataset.at[counter, column] = mean
            counter += 1
    
    new_dataset = TSNE(n_components=2, perplexity=5, n_iter=1000).fit_transform(day_split_dataset)
    new_dataframe = pd.DataFrame(new_dataset, columns=['component_1', 'component_2'])
    # n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
    neighbors = NearestNeighbors(n_neighbors=5).fit(new_dataframe)
    # Find the k-neighbors of a point
    neighbor_distance, neighbor_index = neighbors.kneighbors(new_dataframe)

    sorted_neighbor_distance = np.sort(neighbor_distance, axis=0)

    k_distance = sorted_neighbor_distance[:,4]

    plt.plot(k_distance)
    plt.axhline(y=5, linewidth=1, linestyle='dashed', color='k')
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (4th NN)")
    plt.show()

    clusters = DBSCAN(eps=5, min_samples=4).fit(new_dataframe)
    set(clusters.labels_)
    Counter(clusters.labels_)
    color_list = ['black']
    for n_colors in range(len(set(clusters.labels_)) - 1):
        hexadecimal = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        color_list.append(hexadecimal)

        
    plot = sns.scatterplot(data=new_dataframe, x='component_1', y='component_2', hue=clusters.labels_, legend='full', palette=color_list)
    sns.move_legend(plot, 'upper right', bbox_to_anchor=(1.2, 1.2), title='Clusters')
    plt.show()
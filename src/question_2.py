import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, scale
import os
import numpy
import pandas

# Elbow method of finding appropriate number of clusters
def find_best_clusters(number_of_clusters, scaled_dataset):
    # Testing which number of clusters is appropriate
    sse = []
    
    for k in range(1, number_of_clusters):
        inertia = kmeans_alg(n_clusters=k, n_init=10, max_iter=30, init='k-means++', dataset=scaled_dataset)
        sse.append(inertia)
        
    knee_locator = KneeLocator(range(1, number_of_clusters), sse, curve='convex', direction='decreasing')
    return knee_locator.elbow
  

def kmeans_alg(n_clusters, n_init, max_iter, init, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, init=init)    
    kmeans = KMeans()
    # fit_predict is just a convenience method that calls fit
    label = kmeans.fit_predict(dataset)
    
    return kmeans.inertia_, kmeans.cluster_centers_, kmeans.labels_, label


def preproccess_data(dataset):
    # Number of 5 minute batches in 24 hours
    batch_size = 288
    dataframe = dataset.loc[:,['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    columns = dataframe.columns
    
    day_split_dataset = pandas.DataFrame(columns=columns)
    for column in columns:
        counter = 0
        # Getting data of each column
        data = dataframe.loc[:,column]
        # Spliting data per day 
        mini_batches = [data[k: k + batch_size] for k in range(0, len(data), batch_size)]
        for mini_batch in mini_batches:
            # For every batch and column find mean and place it in new dataset
            mean = numpy.round(mini_batch.mean(), 2)
            day_split_dataset.at[counter, column] = mean
            counter += 1
            
    return day_split_dataset


def handle_data_trial(day_split_dataset):
    # day_split_dataset=day_split_dataset.loc[:,['Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    # dimensionality reduction technique
    pca = PCA(2)
    new_dataset = pca.fit_transform(day_split_dataset)
    # new_dataset = pandas.DataFrame(columns=['Current demand', 'Overall supply'])
    # new_dataset['Current demand'] = day_split_dataset['Current demand']
    
    # columns = day_split_dataset.columns
    # energy_dataset = day_split_dataset['Solar']
    # for column in columns[4:]:
    #     energy_dataset += day_split_dataset[column]
    # new_dataset['Overall supply'] = energy_dataset
    return new_dataset
    

# 288
if __name__ == '__main__':
    path = os.getcwd() + '/dataset/final_dataset.csv'
    dataset = pandas.read_csv(path)
    
    day_split_dataset = preproccess_data(dataset)
    scaled_dataset = handle_data_trial(day_split_dataset=day_split_dataset) 
    # scaled_dataset = day_split_dataset.loc[:,['Current demand', 'Solar', 'Wind', 'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    # scaled_dataset = scaled_dataset.to_numpy()
    # optimal_no_clusters = find_best_clusters(number_of_clusters=30, scaled_dataset=scaled_dataset)
    # print(optimal_no_clusters)
    
    inertia, centroids, labels, label = kmeans_alg(n_clusters=8, n_init=10, max_iter=30, init='k-means++', dataset=scaled_dataset)
    
    # WE need to add all the supply and cluster according to that, we have the demand we need the added supply
    for i in numpy.unique(labels):
        plt.scatter(scaled_dataset[label == i , 0] , scaled_dataset[label == i , 1] , label = i)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color="black")
    plt.legend()
    plt.show()
    
    
    
    
    
    

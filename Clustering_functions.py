import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans

def how_many_clusters(PCA_df, i_cluster, n_cluster, step):
    '''
    input
    output
    '''
    scores = []
    for ii in range(i_cluster, n_cluster, step):
        kmeans = KMeans(n_clusters=ii)
        clf = kmeans.fit(PCA_df)
        score_i = clf.score(PCA_df)
        scores.append(abs(score_i))

    plt.figure(figsize=(16, 8))
    plt.plot(range(i_cluster, n_cluster, step), scores, linestyle='--', marker='h', color='r')
    plt.xlabel('i clusters')
    plt.ylabel('Sum of squared errors')
    plt.title('lose vs. number of clusters')
    plt.savefig("plots/how_many_clusters.png")
    plt.clf()

    return scores

def fit_clusters(PCA_df, n_cluster):
    '''
    input
    output
    '''
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_clf = kmeans.fit(PCA_df)
    cluster_predict = cluster_clf.predict(PCA_df)

    cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns = PCA_df.columns)

    return cluster_clf, cluster_predict, cluster_centers_df

def cluster_proportion(cluster_predict):
    '''
    input
    output
    '''
    cluster_prediction_df = pd.DataFrame(cluster_predict, columns = ['cluster'])
    cluster_tots = cluster_prediction_df['cluster'].value_counts().rename_axis('cluster').reset_index(name='cluster_tot')
    cluster_tots['cluster_percent'] = cluster_tots['cluster_tot'] / len(cluster_predict) * 100
    cluster_tots = cluster_tots.sort_values('cluster')

    return cluster_tots

def plot_cluster_proportion(azdias_cluster_tots, customers_cluster_tots):
    '''
    input
    output
    '''
    plt.bar(x = azdias_cluster_tots['cluster'] -0.2, height = azdias_cluster_tots['cluster_percent'], width = 0.4, label = 'Azdias')
    plt.bar(x = customers_cluster_tots['cluster'] +0.2, height = customers_cluster_tots['cluster_percent'], width = 0.4, label = 'Customers')
    plt.xlabel = 'Cluster'
    plt.ylabel = 'Percentage per Cluster'
    plt.legend()
    plt.savefig("plots/cluster_compare.png")
    plt.clf()

def create_comparison_df(azdias_cluster_tots, customers_cluster_tots):
    '''
    input
    output
    '''
    azdias_cluster_tots = azdias_cluster_tots.add_suffix('_azdias')
    customers_cluster_tots = customers_cluster_tots.add_suffix('_customers')

    cluster_compare = pd.merge(azdias_cluster_tots,
                           customers_cluster_tots,
                           left_on = 'cluster_azdias',
                           right_on = 'cluster_customers'
                           )
    
    cluster_compare = cluster_compare.drop('cluster_customers', axis = 1)
    cluster_compare = cluster_compare.rename(columns = {'cluster_azdias':'cluster'})
    cluster_compare['cluster_prop_diff'] = 2 * (cluster_compare['cluster_percent_customers'] - cluster_compare['cluster_percent_azdias']) / (cluster_compare['cluster_percent_customers'] + cluster_compare['cluster_percent_azdias'])
    
    return cluster_compare

def plot_cluster_compare(cluster_compare):
    '''
    input
    output
    '''
    plt.bar(x = cluster_compare['cluster'], height = cluster_compare['cluster_prop_diff'], width = 0.4)
    plt.xlabel = 'Cluster'
    plt.ylabel = 'Proportional cluster difference'
    plt.savefig("plots/cluster_compare_percent_diff.png")
    plt.clf()
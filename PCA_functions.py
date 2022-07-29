import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_clean(df):
    '''
    input: the cleaned data from step 0 has been saved so we can bring it in for fresh scripts
    output: teh clean dataframe
    '''
    # load
    df_clean = pd.read_csv('data/' + df + '.csv')
    # prepare
    df_clean = df_clean.astype(dtype='float64')
    df_clean = df_clean.drop('Unnamed: 0', axis = 1)

    return df_clean

def create_PCA_and_cumulative_explained_variance(df_clean):
    '''
    input: clean dataframe
    output: I kept adding to this function because I didn't want to repeat steps, so the outputs are in stages.
    They include; pca_df for the transformed dataframe; a datafraem for the acculamulated variance so that 
    we can find a threshold; and feature_weights so that we can sort which componets are strongly correlated. 
    '''
    # need length for maximum number of feature/components
    num_features = df_clean.shape[1]
    # got to be scaled
    df_scaled = StandardScaler().fit_transform(df_clean)
    # use the PCA library function and applt to scaled df
    pca = PCA(n_components = num_features, random_state=42)
    principal_components = pca.fit_transform(df_scaled)

    # create dataframe with appriopriate column names
    PCA_comp_list = []
    for ii in range(0,num_features):
        jj = str(ii+1)
        x = 'PC'+jj
        PCA_comp_list.append(x)

    pca_df = pd.DataFrame(data = principal_components, columns = PCA_comp_list)
    #
    cum_explained_var = []
    for ii in range(0, len(pca.explained_variance_ratio_)):
        if ii == 0:
            cum_explained_var.append(pca.explained_variance_ratio_[ii])
        else:
            cum_explained_var.append(pca.explained_variance_ratio_[ii] + cum_explained_var[ii-1])

    df_cum_explained_var = pd.DataFrame(cum_explained_var)
    df_cum_explained_var = df_cum_explained_var.reset_index()
    df_cum_explained_var = df_cum_explained_var.rename(columns = {'index': 'NO_pca', 0: 'cum_var'})

    # feature weights
    feature_weights = pca.components_
    feature_weights = pd.DataFrame(data = feature_weights, columns = df_clean.columns)
    feature_weights.index = PCA_comp_list

    return pca_df, df_cum_explained_var, feature_weights

def num_components_threshold(cum_var_df, threshold):
    '''
    input: the dataframe of cumulated variance, and teh threshold we're looked for. i.e. 0.8 for 80%
    output
    '''
    # N is the total number of columns in df - define outside and bring in. 
    cum_var = cum_var_df['cum_var']
    N = cum_var.shape[0]
    for ii in range(N):
        if cum_var[ii] >= threshold:
            # print(cum_var[ii])
            return ii+1

def plot_cum_var_plus_threshold(df_cum_explained_var, threshold):
    '''
    input: dataframe of cumulated variance, and threshold
    output: plot of the acculated variance and indication of threshold.
    '''
    n = df_cum_explained_var.shape[0]
    y_vals = df_cum_explained_var['cum_var']
    x_vals = [num for num in range(1,n+1)]

    width = min(n/3, 20)
    height = min(n/4, 16)
    fig, ax = plt.subplots(figsize=(width,height))
    ax.grid(True)
    ax.set_title('PCA Cumulative Variance')
    ax.set_ylabel("Cumulative Variance % Explained")
    ax.set_xlabel('Principal Components')

    # threshold var
    ax.axhline(threshold, color='black', linewidth=2);
    fig = sns.barplot(x=x_vals, y=y_vals, ax=ax)
    # don't save it was mucking up my later plots. 
    # fig.savefig("./plots/PCA_cum_var.png") 
    return fig

def create_PCA_with_threshold(df_clean, N_features):
    '''
    input: clean dataframe and the number of columns we're taking through to clustering.
    output: this is done in stages as well to create the PCA data but with the readable column names
    which makes understanding the feature weights easier.
    '''
    # need length for maximum number of feature/components
    num_features = N_features
    # got to be scaled
    df_scaled = StandardScaler().fit_transform(df_clean)
    # use the PCA library function and applt to scaled df
    pca = PCA(n_components = num_features, random_state=42)
    principal_components = pca.fit_transform(df_scaled)

    # create dataframe with appriopriate column names
    PCA_comp_list = []
    for ii in range(0,num_features):
        jj = str(ii+1)
        x = 'PC'+jj
        PCA_comp_list.append(x)

    pca_df = pd.DataFrame(data = principal_components, columns = PCA_comp_list)

    # feature weights
    feature_weights = pca.components_
    feature_weights = pd.DataFrame(data = feature_weights, columns = df_clean.columns)
    feature_weights.index = PCA_comp_list

    return pca_df, feature_weights



def explore_feature_weights(feature_weights, ii, n):
    '''
    input: the feature weights per componenant (ii) but only take the top/bottom (n) amount to show
    output: view of teh most imporant feature weight for a given component.
    '''

    top = feature_weights.iloc[ii].sort_values(ascending=False)[:n]
    bottom = feature_weights.iloc[ii].sort_values(ascending=False)[-n:]

    return top, bottom




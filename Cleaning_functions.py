
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# NaN isn't the only way of representing missing values the supporting excel documents tell us that [xx, -1, -1,9] are also being used so we need to map them so they get captured in the missing value analysis.

def collate_unknown(df):
    '''
    Input: Raw dataframe
    Output: Dataframe where other consitions for unknwons have been mapped to np.NaN
    '''
    # create a dictionary with the inout dataset data types
    df_dtypes = dict()
    for col in df.columns:
        df_dtypes[col] = str(df[col].dtype)
    
    # convert to strings so we don't get tripped up by different formatting
    df.astype(dtype='str')
    
    # map for each feature
    for col in df.columns:
        try:
            df.loc[df[col].isin(['X', 'XX', '-1', '-1,9', '-1,0']), col] = np.nan
        except:
            print(col)
    
    return df


def display_NaN(df):
    '''
    input: any dataframe
    output: dataframe containing the number and percentage of NaNs per column
    '''
    df_summary_NaN = pd.DataFrame(columns = ['NaN sum', 'NaN %'])
    df_summary_NaN['NaN sum'] = df.isnull().sum() 
    df_summary_NaN['NaN %'] = df.isnull().sum() / len(df)*100 
    df_summary_NaN = df_summary_NaN.sort_values(['NaN sum'], ascending=False)

    return df_summary_NaN

def column_dropper(df, cols_to_drop):
    '''
    input: dataframe and list of columns to drop
    output: dataframe with columns removed
    '''
    df = df.drop(cols_to_drop, axis=1)
    return df


def row_dropper(df, col):
    '''
    input: dataframe and a given column
    output: datafraem where the rows that weren't populated for the given column have been removed
    '''
    df = df.dropna(subset = [col])
    return df

def test_impute(feature, df):
    '''
    input: dataframe and column to explore
    Output: dataframe with additional columns for mode and median transformations
    '''
    median_col_name = feature + '_median' 
    mode_col_name = feature + '_mode'

    df[median_col_name] = df[feature].fillna(df[feature].median())
    df[mode_col_name] = df[feature].fillna(df[feature].mode())

    return df

def test_impute_plot(feature, df):
    '''
    input: data frame and column to explore
    output: plot of column, median, and mode imputed distributions
    '''
    df_test = test_impute(feature, df)

    nrows = 1
    ncols = 3

    median_col_name = feature + '_median' 
    mode_col_name = feature + '_mode'

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize = (16,4), squeeze=False)
    fig.suptitle('Exposure')

    plt.subplot(1,3,1)
    df_test[feature].hist(bins=50)
    plt.title(feature)

    plt.subplot(1,3,2)
    df_test[median_col_name].hist(bins=50)
    plt.title(median_col_name)

    plt.subplot(1,3,3)
    df_test[mode_col_name].hist(bins=50)
    plt.title(mode_col_name)

    plt.savefig("plots/" + feature + "_impute_compare.png")
    plt.clf()
    return fig

def impute_all_cols(df, strategy):
    '''
    input: dataframe to be imputed with imputed defined by strategy
    output: dataframe with applied Simple Imputer for all columns with NaN, strategy can be specified - but for this project we'll use mode/ most frequent
    '''
    
    imputer = SimpleImputer(strategy=strategy)
    df_trans = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(df_trans, columns=df.columns)

    return imputed_df


def returnNotMatches(df_a,df_b):
    '''
    input: two dataframes
    output: the columns missing or extra between them
    '''
    a = list(df_a.columns)
    b = list(df_b.columns)

    return [[x for x in a if x not in b], [x for x in b if x not in a]]


def encode_cat(df, cat_columns):
    '''
    input: imputed dataframe and list of categorical features
    output: datafraem with encoded categorical features
    '''
    df_cat = df[cat_columns]
    df_cat = pd.get_dummies(df_cat)

    df_clean = pd.concat([df, df_cat], axis=1)
    df_clean = df_clean.drop(cat_columns, axis=1)

    return df_clean


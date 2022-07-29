# Bertelsmann Classifier Project
## Udacity Capstone Project

## Installation
* pandas
* numpy
* matplotlib
* scikit-learn
* XGBoost
* imblearn
* pickle
* LogisticRegression

## Project Motivation

The project is designed to cover data cleaning and preparation for modelling; a customer segmentation analysis using PCA analysis and cluster; ML methods of your choosing to take a conversion model with a class imbalance; and to test your model blindly a TEST set for a Kaggle Competition.

The data is provided by Bertelsmann and will be removed from my device upon completion and submittion of the project. 

|Script| Description |
|-------|----------|
|0_Clean_data.ipynb| The project has four datasets that need to be explored and cleaned so they can be modeled and compared. This script outlines that process and saves clean data at the end.|
|1_Customer_segmentation| This scripts applied the PCA transformation and works out a suitable number of factors that cen be brought forward for clustering and further modelling. It also analyses the suitable number of clusters to use and the clustering analysis was performed for 16 clusters and saved.|
|2_Supervised_learning_model| This script explores class imbalance and model tuing for an XGBoost tree model. The model tuning ran for 2 and a half days and the best parameters model was saved. |
|3_model_analysis| The previuos script generate an .md results file for the model tuning so those metrics can be used in the analysis. This script does further analysis using the validation and the Test dataset. I also bring in some model metrics form teh last stage which was Logistic Regression.  |
|4_Logistic_regress| Even with steps to aid the class imbalance the XGBoost model response distribution is poor and struggles to predict any one that would given a positive response. This script explores balancing using Logistic Regression.|

|Function file| description|
|------|------|
|Cleaning_functions.py| Collates NaNs, cleans and imputes all datasets consistently.|
|PCA_functions.py| All fucntions for the PCA analysis|
|Clustering_fuctions.py| All function for clustering and comparisons between customer and demographic.|
|Modeling_functions.py| All functions for splitting data and running model tuning an dcompiling a results document.|

## results

Results and discussion can be found in the Companion_guide.md and results for the model hyper paramter tuning can be found in XGBoost_test_1.md. All pickled models can be found in /Results.


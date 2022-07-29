import pandas as pd
import os
import numpy as np
import json
import imblearn
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import ParameterGrid 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle

def data_prep(df_clean, cols, response_var, SMOTE_value):
    '''
    input: clean training set, the response to model for and how much to sample up for the class imbalance. 
    output: processed X and y data for training and validation. I do a typical 80-20 split.
    '''
    # if sample_size == 'max':
    #     df_clean = df_clean
    # else:
    #     try:
    #         df_clean = df_clean.sample(sample_size, random_state = 42)
    #     except:
    #         print("The full dataset has been loaded. If this is not what you intented you may have entered an incorrect sample_size. Make sure your input is an integer and not larger than the number of rows in the dataset.")

    # might need to change this to columns to keep 
    df_clean = df_clean.astype(dtype='float64')
    ds_model = df_clean[cols]
    ds_X = ds_model.drop(response_var, axis=1)

    y0 = df_clean[response_var]
    y0 = y0.astype('int')

    # class imbalance
    oversample = SMOTE(sampling_strategy = SMOTE_value)
    smote_ds_X, smote_y0 = oversample.fit_resample(ds_X, y0)

    train_set, validate_set, y_train, y_validate = train_test_split(smote_ds_X, smote_y0, test_size=0.2, random_state=42)

    df_X = train_set
    df_y = y_train

    return df_X, df_y, validate_set, y_validate



'''
model tuning
'''
def model_fit(class_tech_sc,df_X, df_y):
    '''
    input: the model definition and the training data
    output: best model from
    '''
    best_class_tech_sc = class_tech_sc.fit(df_X,df_y)
    return best_class_tech_sc

'''
post modelling analysis
output: model metrics for best model
'''
def min_log_loss_index_params(best_class_tech_sc):

    minLL_index = [i for i, n in enumerate(best_class_tech_sc.cv_results_['mean_test_score']) if n == min(best_class_tech_sc.cv_results_['mean_test_score'])][0]
    minLL_params = best_class_tech_sc.cv_results_['params'][minLL_index]
    
    return minLL_index, minLL_params

'''
Best modellling parameters
output: paramters used in best model
'''
def optimial_params(best_class_tech_sc):
  opt_params = best_class_tech_sc.best_params_

  return opt_params

'''
predictions
output: predictions from best model with test metrics
'''

def predictions(VARS_TEST, best_class_tech_sc, y_TEST):

    model_predictions = best_class_tech_sc.predict_proba(VARS_TEST)
    non_conversion_rate = model_predictions[:,0]
    conversion_rate = model_predictions[:,1]

    neg_log_loss_test = -abs(log_loss(y_TEST, conversion_rate))

    return non_conversion_rate, conversion_rate, neg_log_loss_test

'''
feature importance
output: top 20 features from model
'''
def feature_importance(df_X, best_class_tech_sc):

    importances = best_class_tech_sc.best_estimator_.feature_importances_
    names = df_X.columns

    d = {'feature name': names, 'importance': importances}
    importance_df = pd.DataFrame(data=d)
    importance_df_sort = importance_df.sort_values(by=['importance'], ascending=False).round(3)

    return importance_df_sort


'''
Reliability Graph
output: model metric - reliability graph
'''
def graphic_reliability(VARS_VALIDATE, best_class_tech_sc, y_VALIDATE, results_file_name):

    non_conversion_rate, conversion_rate, neg_log_loss_test = predictions(VARS_VALIDATE, best_class_tech_sc, y_VALIDATE)
    fop, mpv = calibration_curve(y_VALIDATE, conversion_rate, n_bins=10, normalize = False)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(mpv, fop, marker='.')
    plt.savefig('plots/' + results_file_name + '.jpg')
    plt.close

    return print("graphic saved")

'''
some json object formating so I can display it in a mk file
'''
INDENT = 2
SPACE = " "
NEWLINE = "\n"

# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0):
  ret = ""
  if isinstance(o, dict):
    ret += "{" + NEWLINE
    comma = ""
    for k, v in o.items():
      ret += comma
      comma = ",\n"
      ret += SPACE * INDENT * (level + 1)
      ret += '"' + str(k) + '":' + SPACE
      ret += to_json(v, level + 1)

    ret += NEWLINE + SPACE * INDENT * level + "}"
  elif isinstance(o, str):
    ret += '"' + o + '"'
  elif isinstance(o, list):
    ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
  # Tuples are interpreted as lists
  elif isinstance(o, tuple):
    ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
  elif isinstance(o, bool):
    ret += "true" if o else "false"
  elif isinstance(o, int):
    ret += str(o)
  elif isinstance(o, float):
    ret += '%.7g' % o
  elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
    ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
  elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
    ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
  elif o is None:
    ret += 'null'
  else:
    raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
  return ret


'''
create text file for results of model
output: compile a mk file with what you need to know about the model tuning you've just done.
'''
def results_to_text(best_class_tech_sc, VARS_VALIDATE, y_VALIDATE, df_X, param_grid, results_file_name):

    minLL_index, minLL_params = min_log_loss_index_params(best_class_tech_sc)

    opt_params = optimial_params(best_class_tech_sc)

    minLL_params = to_json(minLL_params)
    opt_params = to_json(opt_params)
    pretty_param_grid = to_json(param_grid)

    MtrainS = max(best_class_tech_sc.cv_results_['mean_train_score'])
    MtestS = max(best_class_tech_sc.cv_results_['mean_test_score'])

    non_conversion_rate, conversion_rate, neg_log_loss_test = predictions(VARS_VALIDATE, best_class_tech_sc, y_VALIDATE)
    
    importance_df_sort = feature_importance(df_X, best_class_tech_sc)

    top_20_important = importance_df_sort.head(20)

    file = open('Results/' + results_file_name + '.md', "w+")
    
    file.write(
f"""# Model tuning best results
Results file for best modelling results for trial `{results_file_name}`.'

---

The tuning parameter grid:

```json
{pretty_param_grid}
```

The best_params_ grid from function is 
```json
{opt_params}
```

The max mean train score is {MtrainS}.

The max mean test score is {MtestS}.

The model predicted neg log loss on TEST is {neg_log_loss_test}.

---

The reliabilty diagram:
![reliability graphic](./{results_file_name}.jpg)

---

Top 20 features by importance:

```python
{top_20_important}
```

"""
    )
    return print("results file created")

'''
pickle dump that model
input: best model
output: pickled model and saved in results
'''
def save_model(best_class_tech_sc, results_file_name):
  filename = 'Results/' + results_file_name + '.pkl'
  pickle.dump(best_class_tech_sc, open(filename, 'wb'))
  return print('model pickled')
import shap

from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

# Magnet  nebor 200 gamma =0 dim=64
# GAT nebor 50 gamma 0 dim 64
# Graph SAGE nebor 50 gamma 2 dim 32







model_type = 'Magnet'
model_name = model_type+'_cluster'
data = pd.read_csv('Magnet k=3 data.csv')

output_path = 'Predict_model/' + model_type + ' k=3'+'/'
print(data.shape)
def compute_macro_roc(y_true, y_pred_prob):
    """
    Computes macro-average ROC curve for multi-class data.

    Parameters:
    - y_true: Ground truth (true) labels.
    - y_pred_prob: Prediction probabilities.

    Returns:
    - fpr_macro: Macro-average false positive rate.
    - tpr_macro: Macro-average true positive rate.
    - auroc_macro: Area under the macro-average ROC curve.
    """
    n_classes = len(np.unique(y_true))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auroc_macro = auc(fpr_macro, tpr_macro)

    return fpr_macro, tpr_macro, auroc_macro



# select features

features = data.iloc[:, 1:-1].copy()
target = data[model_name]
target_label = model_name


X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target, 
    test_size=0.2, 
    random_state=42, 
    stratify=target  # Ensures stratified splitting
)

# over sampling
ros = RandomOverSampler(random_state=42)
X_train_overampled, y_train_oversampled = ros.fit_resample(X_train, y_train)

# Check the distribution of the classes after resampling
print("Class distribution after over resampling:", dict(zip(*np.unique(y_train_oversampled, return_counts=True))))

# under sampling
rus = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

# Check the distribution of the classes after resampling
print("Class distribution after under resampling:", dict(zip(*np.unique(y_train_undersampled, return_counts=True))))

imp = SimpleImputer( strategy='mean')

#lrc = linear_model.LogisticRegression()
lrc = linear_model.LogisticRegression(multi_class='multinomial')

#xgb = XGBClassifier()
number_of_classes = len(np.unique(y_test))
xgb = XGBClassifier(objective='multi:softprob', num_class=number_of_classes)

model_1 = Pipeline([
    ('imputation', imp),
    ('classifier', lrc)
])

model_2 = Pipeline([
    ('imputation', imp),
    ('classifier', xgb)
])

from sklearn.model_selection import GridSearchCV

# Logistic regression parameter search space (grid search)
model_1_parameters = {
    "classifier__C": [0.01, 0.1,],  
    # "classifier__solver": ["newton-cg", "lbfgs", "saga"],  
    # "classifier__penalty": ["none", "l2", "elasticnet"],
    "classifier__l1_ratio": [0.1, 0.5, 0.9]  # Only applicable for elasticnet
}


# XGBoost parameter search space (grid search)
model_2_parameters = {
    "classifier__learning_rate": [0.01,  0.2],
    "classifier__gamma": [0.01,  1],  
    # "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    # "classifier__max_depth": [3, 4, 5, 6, 7, 8, 9],
    # "classifier__min_child_weight": [0.1, 0.5, 1],  
    # "classifier__gamma": [0.01, 0.1, 1],  
    # "classifier__subsample": [0.5, 0.7, 0.9],  
    # "classifier__colsample_bytree": [0.5, 0.7, 0.9],
    # "classifier__reg_alpha": [0.01, 0.1, 1],  
    # "classifier__reg_lambda": [0.01, 0.1, 1],  
    # "classifier__n_estimators": [50, 100, 200],
    # "classifier__scale_pos_weight": [1, 5, 10]
}


# Create GridSearchCV
model_3 = GridSearchCV(model_1, model_1_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)
model_4 = GridSearchCV(model_2, model_2_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)

# Train models (original data)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)

# Train models (oversampled data)
model_5 = GridSearchCV(model_1, model_1_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)
model_6 = GridSearchCV(model_2, model_2_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)
model_5.fit(X_train_overampled, y_train_oversampled)
model_6.fit(X_train_overampled, y_train_oversampled)

# Train models (undersampled data)
model_7 = GridSearchCV(model_1, model_1_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)
model_8 = GridSearchCV(model_2, model_2_parameters, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=3)
model_7.fit(X_train_undersampled, y_train_undersampled)
model_8.fit(X_train_undersampled, y_train_undersampled)

joblib.dump(model_3.best_estimator_, output_path + 'model_1_grid.joblib')
joblib.dump(model_4.best_estimator_, output_path + 'model_2_grid.joblib')
joblib.dump(model_5.best_estimator_, output_path + 'model_3_grid.joblib')
joblib.dump(model_6.best_estimator_, output_path + 'model_4_grid.joblib')
joblib.dump(model_7.best_estimator_, output_path + 'model_5_grid.joblib')
joblib.dump(model_8.best_estimator_, output_path + 'model_6_grid.joblib')

# View best parameters
print("Best parameters for Logistic Regression (Grid Search):", model_3.best_params_)
print("Best parameters for XGBoost (Grid Search):", model_4.best_params_)



model_3_predict_prob = model_3.predict_proba(X_test)
model_4_predict_prob = model_4.predict_proba(X_test)
model_5_predict_prob = model_5.predict_proba(X_test)
model_6_predict_prob = model_6.predict_proba(X_test)
model_7_predict_prob = model_7.predict_proba(X_test)
model_8_predict_prob = model_8.predict_proba(X_test)

# Use the function for model_1 and model_2
model_3_fpr_macro, model_3_tpr_macro, model_3_auroc_macro = compute_macro_roc(y_test, model_3_predict_prob)
model_4_fpr_macro, model_4_tpr_macro, model_4_auroc_macro = compute_macro_roc(y_test, model_4_predict_prob)

model_5_fpr_macro, model_5_tpr_macro, model_5_auroc_macro = compute_macro_roc(y_test, model_5_predict_prob)
model_6_fpr_macro, model_6_tpr_macro, model_6_auroc_macro = compute_macro_roc(y_test, model_6_predict_prob)


model_7_fpr_macro, model_7_tpr_macro, model_7_auroc_macro = compute_macro_roc(y_test, model_7_predict_prob)
model_8_fpr_macro, model_8_tpr_macro, model_8_auroc_macro = compute_macro_roc(y_test, model_8_predict_prob)




model_3_auroc_macro = round(model_3_auroc_macro, 2)
model_4_auroc_macro = round(model_4_auroc_macro, 2)
model_5_auroc_macro = round(model_5_auroc_macro, 2)
model_6_auroc_macro = round(model_6_auroc_macro, 2)
model_7_auroc_macro = round(model_7_auroc_macro, 2)
model_8_auroc_macro = round(model_8_auroc_macro, 2)


# Now, you can plot the macro-average ROC curves:
plt.figure(figsize=(30,20))
ax = plt.subplot(1,3,1)
ax.plot([0, 1], [0, 1], 'k--*')

ax.plot(model_3_fpr_macro, model_3_tpr_macro, label='Logistic Regression (grid_search) -> ' + str(round(model_3_auroc_macro, 2)))
ax.plot(model_4_fpr_macro, model_4_tpr_macro, label='XGBoost (grid_search) -> ' + str(round(model_4_auroc_macro, 2)))
ax.plot(model_5_fpr_macro, model_5_tpr_macro, label='Logistic Regression (Oversampled + grid_search) -> ' + str(round(model_5_auroc_macro, 2)))
ax.plot(model_6_fpr_macro, model_6_tpr_macro, label='XGBoost (Oversampled + grid_search) -> ' + str(round(model_6_auroc_macro, 2)))
ax.plot(model_7_fpr_macro, model_7_tpr_macro, label='Logistic Regression (Undersampled + grid_search) -> ' + str(round(model_7_auroc_macro, 2)))
ax.plot(model_8_fpr_macro, model_8_tpr_macro, label='XGBoost (Undersampled + grid_search) -> ' + str(round(model_8_auroc_macro, 2)))
ax.legend()
plt.savefig(output_path +'macro_roc_curves.png')
ax.set_aspect(1)

print("model 1 summary: \n", classification_report(y_test, model_3.predict(X_test)))
print("model 2 summary: \n", classification_report(y_test, model_4.predict(X_test)))
print("model 3 summary: \n", classification_report(y_test, model_5.predict(X_test)))
print("model 4 summary: \n", classification_report(y_test, model_6.predict(X_test)))
print("model 5 summary: \n", classification_report(y_test, model_7.predict(X_test)))
print("model 6 summary: \n", classification_report(y_test, model_8.predict(X_test)))


# shap start
import pandas as pd
import os
import shap

from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
k=3
data = pd.read_csv(f'{model_type} k=3 data.csv')
output_path = f'shap_result/{model_type} k=3'
os.makedirs(output_path, exist_ok=True)
features = data.iloc[:, 1:-1].copy()
model_name = model_type+'_cluster'
target = data[model_name]
target_label = model_name


X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target, 
    test_size=0.2, 
    random_state=42, 
    stratify=target  # Ensures stratified splitting
)

from joblib import load
model = model_type
# Load the models
model_3 = load(f'Predict_model/{model} k={k}/model_1_grid.joblib')
model_4 = load(f'Predict_model/{model} k={k}/model_2_grid.joblib')
model_5 = load(f'Predict_model/{model} k={k}/model_3_grid.joblib')
model_6 = load(f'Predict_model/{model} k={k}/model_4_grid.joblib')
model_7 = load(f'Predict_model/{model} k={k}/model_5_grid.joblib')
model_8 = load(f'Predict_model/{model} k={k}/model_6_grid.joblib')



import shap

from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
def compute_macro_roc(y_true, y_pred_prob):
    """
    Computes macro-average ROC curve for multi-class data.

    Parameters:
    - y_true: Ground truth (true) labels.
    - y_pred_prob: Prediction probabilities.

    Returns:
    - fpr_macro: Macro-average false positive rate.
    - tpr_macro: Macro-average true positive rate.
    - auroc_macro: Area under the macro-average ROC curve.
    """
    n_classes = len(np.unique(y_true))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auroc_macro = auc(fpr_macro, tpr_macro)

    return fpr_macro, tpr_macro, auroc_macro

model_3_predict_prob = model_3.predict_proba(X_test)
model_4_predict_prob = model_4.predict_proba(X_test)
model_5_predict_prob = model_5.predict_proba(X_test)
model_6_predict_prob = model_6.predict_proba(X_test)
model_7_predict_prob = model_7.predict_proba(X_test)
model_8_predict_prob = model_8.predict_proba(X_test)

# Use the function for model_1 and model_2
model_3_fpr_macro, model_3_tpr_macro, model_3_auroc_macro = compute_macro_roc(y_test, model_3_predict_prob)
model_4_fpr_macro, model_4_tpr_macro, model_4_auroc_macro = compute_macro_roc(y_test, model_4_predict_prob)

model_5_fpr_macro, model_5_tpr_macro, model_5_auroc_macro = compute_macro_roc(y_test, model_5_predict_prob)
model_6_fpr_macro, model_6_tpr_macro, model_6_auroc_macro = compute_macro_roc(y_test, model_6_predict_prob)


model_7_fpr_macro, model_7_tpr_macro, model_7_auroc_macro = compute_macro_roc(y_test, model_7_predict_prob)
model_8_fpr_macro, model_8_tpr_macro, model_8_auroc_macro = compute_macro_roc(y_test, model_8_predict_prob)




model_3_auroc_macro = round(model_3_auroc_macro, 2)
model_4_auroc_macro = round(model_4_auroc_macro, 2)
model_5_auroc_macro = round(model_5_auroc_macro, 2)
model_6_auroc_macro = round(model_6_auroc_macro, 2)
model_7_auroc_macro = round(model_7_auroc_macro, 2)
model_8_auroc_macro = round(model_8_auroc_macro, 2)


# Now, you can plot the macro-average ROC curves:
plt.figure(figsize=(15,15))
ax = plt.subplot(1,1,1)
ax.plot([0, 1], [0, 1], 'k--*')

ax.plot(model_3_fpr_macro, model_3_tpr_macro, label='Logistic Regression -> ' + str(round(model_3_auroc_macro, 2)))
ax.plot(model_4_fpr_macro, model_4_tpr_macro, label='XGBoost -> ' + str(round(model_4_auroc_macro, 2)))
ax.plot(model_5_fpr_macro, model_5_tpr_macro, label='Logistic Regression (Oversampled) -> ' + str(round(model_5_auroc_macro, 2)))
ax.plot(model_6_fpr_macro, model_6_tpr_macro, label='XGBoost (Oversampled) -> ' + str(round(model_6_auroc_macro, 2)))
ax.plot(model_7_fpr_macro, model_7_tpr_macro, label='Logistic Regression (Undersampled) -> ' + str(round(model_7_auroc_macro, 2)))
ax.plot(model_8_fpr_macro, model_8_tpr_macro, label='XGBoost (Undersampled) -> ' + str(round(model_8_auroc_macro, 2)))
ax.legend(loc='lower right',fontsize = 23)

ax.set_aspect(1)
# ax.set_xlabel(fontsize=20)  # Set x-axis title and font size
# ax.set_ylabel(fontsize=20)  #
ax.tick_params(axis='both', which='major', labelsize=28)  #
plt.savefig(output_path +'/macro_roc_curves.png')

print("model 1 summary: \n", classification_report(y_test, model_3.predict(X_test), digits=3))
print("model 2 summary: \n", classification_report(y_test, model_4.predict(X_test), digits=3))
print("model 3 summary: \n", classification_report(y_test, model_5.predict(X_test), digits=3))
print("model 4 summary: \n", classification_report(y_test, model_6.predict(X_test), digits=3))
print("model 5 summary: \n", classification_report(y_test, model_7.predict(X_test), digits=3))
print("model 6 summary: \n", classification_report(y_test, model_8.predict(X_test), digits=3))

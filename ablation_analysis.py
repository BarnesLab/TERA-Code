from pathlib import Path
import os
import random
import json
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

DATA_ROOT=Path('TERA/TERA Analysis')
PROJECT_ROOT=Path('TERA')
RESULTS=PROJECT_ROOT/'results'
RESULTS.mkdir(parents=True,exist_ok=True)

DAILY_DATA=DATA_ROOT/'exploratory_analysis/tera_daily.csv'
WEEKLY_DATA=DATA_ROOT/'exploratory_analysis/weekly/tera_weekly.csv'

print('TensorFlow:',tf.__version__)
print('Daily data:',DAILY_DATA)
print('Weekly data:',WEEKLY_DATA)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def calculate_metrics(y_true,probability,threshold=0.50):
    prediction=(np.asarray(probability)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,prediction,labels=[0,1]).ravel()
    return {
        'accuracy_pct':100*accuracy_score(y_true,prediction),
        'precision_pct':100*precision_score(y_true,prediction,zero_division=0),
        'recall_pct':100*recall_score(y_true,prediction,zero_division=0),
        'specificity_pct':100*tn/(tn+fp) if (tn+fp)>0 else np.nan,}

def build_model(mode,steps,n_dynamic,n_static):
    inputs=[]
    representations=[]

    if mode in ('dynamic','integrated'):
        dynamic_input=tf.keras.Input((steps,n_dynamic),name='dynamic')
        dynamic_representation=tf.keras.layers.LSTM(128,return_sequences=True)(dynamic_input)
        dynamic_representation=tf.keras.layers.LSTM(128)(dynamic_representation)
        inputs.append(dynamic_input)
        representations.append(dynamic_representation)

    if mode in ('static','integrated'):
        static_input=tf.keras.Input((n_static,),name='static')
        static_representation=tf.keras.layers.Dense(32,activation='relu')(static_input)
        inputs.append(static_input)
        representations.append(static_representation)

    x=representations[0] if len(representations)==1 else tf.keras.layers.Concatenate()(representations)
    x=tf.keras.layers.Dropout(0.25)(x)
    x=tf.keras.layers.Dense(32,activation='relu')(x)
    output=tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model=tf.keras.Model(inputs=inputs,outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='binary_crossentropy')
    return model

MODEL_MODES={
    'Dynamic-only (EDM)':'dynamic',
    'Static-only':'static',
    'Integrated (all modalities)':'integrated',}


DAILY_SEED=20260811
daily=pd.read_csv(DAILY_DATA).dropna(subset=['USUBJID','withinrange']).reset_index(drop=True)
daily['USUBJID']=daily.USUBJID.astype(int)
daily_y=daily.withinrange.astype(int).to_numpy()
daily_groups=daily.USUBJID.to_numpy()

daily_features=['is_Weekend','is_adherent','Morning','Afternoon','Evening','Night']
daily_dynamic_columns=[f't-{lag} {feature}' for lag in range(7,0,-1) for feature in daily_features]
daily_excluded={'USUBJID','withinrange','ADY','is_Weekend','Morning','Afternoon','Evening','Night','is_adherent'}
daily_static_columns=[c for c in daily.columns if c not in daily_excluded and c not in daily_dynamic_columns]

daily_dynamic=daily[daily_dynamic_columns].apply(pd.to_numeric,errors='coerce').to_numpy(float).reshape(-1,7,len(daily_features))
daily_static=daily[daily_static_columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)

print('Daily observations:',len(daily))
print('Daily participants:',daily.USUBJID.nunique())
print('Dynamic shape:',daily_dynamic.shape)
print('Static shape:',daily_static.shape)
print('Static features:',len(daily_static_columns))

def run_daily_models():
    set_seed(DAILY_SEED)
    predictions={name:np.full(len(daily),np.nan) for name in MODEL_MODES}
    training_log=[]
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=DAILY_SEED)

    for fold,(train_all,test) in enumerate(outer.split(np.zeros(len(daily_y)),daily_y,daily_groups),1):
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=DAILY_SEED+fold-1)
        train_relative,validation_relative=next(splitter.split(train_all,daily_y[train_all],daily_groups[train_all]))
        train=train_all[train_relative]
        validation=train_all[validation_relative]

        flat=daily_dynamic.reshape(len(daily),-1)
        dynamic_imputer=SimpleImputer(strategy='median').fit(flat[train])
        dynamic_scaler=StandardScaler().fit(dynamic_imputer.transform(flat[train]))
        def transform_dynamic(index):
            values=dynamic_scaler.transform(dynamic_imputer.transform(flat[index]))
            return values.reshape(-1,7,len(daily_features)).astype('float32')

        static_imputer=SimpleImputer(strategy='median').fit(daily_static[train])
        static_scaler=StandardScaler().fit(static_imputer.transform(daily_static[train]))
        def transform_static(index):
            return static_scaler.transform(static_imputer.transform(daily_static[index])).astype('float32')

        dtrain,dvalidation,dtest=[transform_dynamic(i) for i in (train,validation,test)]
        strain,svalidation,stest=[transform_static(i) for i in (train,validation,test)]

        positives=daily_y[train].sum()
        negatives=len(train)-positives
        class_weights={0:len(train)/(2*negatives),1:len(train)/(2*positives)}

        for model_name,mode in MODEL_MODES.items():
            tf.keras.backend.clear_session()
            tf.random.set_seed(DAILY_SEED+fold-1)
            model=build_model(mode,7,len(daily_features),strain.shape[1])

            if mode=='dynamic':
                xtrain,xvalidation,xtest=dtrain,dvalidation,dtest
            elif mode=='static':
                xtrain,xvalidation,xtest=strain,svalidation,stest
            else:
                xtrain=[dtrain,strain]
                xvalidation=[dvalidation,svalidation]
                xtest=[dtest,stest]

            early_stopping=tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',patience=3,min_delta=1e-4,restore_best_weights=True)
            history=model.fit(
                xtrain,daily_y[train],validation_data=(xvalidation,daily_y[validation]),
                epochs=30,batch_size=64,class_weight=class_weights,
                callbacks=[early_stopping],verbose=0)
            predictions[model_name][test]=model.predict(xtest,batch_size=256,verbose=0).ravel()
            training_log.append({'fold':fold,'model':model_name,'epochs':len(history.history['loss'])})
            print(f'Daily fold {fold}: {model_name}; epochs={len(history.history["loss"])}')

    rows=[{'model':name,**calculate_metrics(daily_y,p)} for name,p in predictions.items()]
    return pd.DataFrame(rows),predictions,pd.DataFrame(training_log)

RUN_DAILY=True  # Set to False to load the saved verified metrics without retraining.
if RUN_DAILY:
    daily_results,daily_predictions,daily_training_log=run_daily_models()
    daily_results.to_csv(RESULTS/'daily_ablation_metrics.csv',index=False)
    pd.DataFrame({'USUBJID':daily_groups,'outcome':daily_y,**daily_predictions}).to_csv(RESULTS/'daily_three_model_predictions.csv',index=False)
    daily_training_log.to_csv(RESULTS/'daily_three_model_training_log.csv',index=False)
else:
    daily_results=pd.read_csv(RESULTS/'daily_ablation_metrics.csv')

print(daily_results.round(2).to_string(index=False))


WEEKLY_SEED=20260818
weekly=pd.read_csv(WEEKLY_DATA).dropna(subset=['USUBJID','is_adherent_wk']).reset_index(drop=True)
weekly['USUBJID']=weekly.USUBJID.astype(int)
weekly_y=weekly.is_adherent_wk.astype(int).to_numpy()
weekly_groups=weekly.USUBJID.to_numpy()

weekly_features=['weekend_adherent','epoch_time1_mode_agg','time_mean','time_std','is_adherent_wk']
weekly_dynamic_columns=[f't-{lag} {feature}' for lag in range(4,0,-1) for feature in weekly_features]
weekly_excluded={'USUBJID','is_adherent_wk','week_num','weekend_adherent','time_mean','time_std','adherent_percent','epoch_time1_mode_agg'}
weekly_static_columns=[c for c in weekly.columns if c not in weekly_excluded and c not in weekly_dynamic_columns]

weekly_dynamic=weekly[weekly_dynamic_columns].apply(pd.to_numeric,errors='coerce').to_numpy(float).reshape(-1,4,len(weekly_features))
weekly_static=weekly[weekly_static_columns].copy()

print('Weekly observations:',len(weekly))
print('Weekly participants:',weekly.USUBJID.nunique())
print('Dynamic shape:',weekly_dynamic.shape)
print('Static shape:',weekly_static.shape)
print('Static features:',len(weekly_static_columns))

def run_weekly_models():
    set_seed(WEEKLY_SEED)
    predictions={name:np.full(len(weekly),np.nan) for name in MODEL_MODES}
    training_log=[]
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=WEEKLY_SEED)

    for fold,(train_all,test) in enumerate(outer.split(np.zeros(len(weekly_y)),weekly_y,weekly_groups),1):
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=WEEKLY_SEED+fold-1)
        train_relative,validation_relative=next(splitter.split(train_all,weekly_y[train_all],weekly_groups[train_all]))
        train=train_all[train_relative]
        validation=train_all[validation_relative]

        flat=weekly_dynamic.reshape(len(weekly),-1)
        dynamic_imputer=SimpleImputer(strategy='median').fit(flat[train])
        dynamic_scaler=StandardScaler().fit(dynamic_imputer.transform(flat[train]))
        def transform_dynamic(index):
            values=dynamic_scaler.transform(dynamic_imputer.transform(flat[index]))
            return values.reshape(-1,4,len(weekly_features)).astype('float32')

        categorical=[c for c in weekly_static_columns if weekly_static[c].dtype=='object']
        numeric=[c for c in weekly_static_columns if c not in categorical]
        static_preprocessor=ColumnTransformer([
            ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric),
            ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical),
        ]).fit(weekly_static.iloc[train])

        dtrain,dvalidation,dtest=[transform_dynamic(i) for i in (train,validation,test)]
        strain,svalidation,stest=[static_preprocessor.transform(weekly_static.iloc[i]).astype('float32') for i in (train,validation,test)]

        positives=weekly_y[train].sum()
        negatives=len(train)-positives
        class_weights={0:len(train)/(2*negatives),1:len(train)/(2*positives)}

        for model_name,mode in MODEL_MODES.items():
            tf.keras.backend.clear_session()
            tf.random.set_seed(WEEKLY_SEED+fold-1)
            model=build_model(mode,4,len(weekly_features),strain.shape[1])

            if mode=='dynamic':
                xtrain,xvalidation,xtest=dtrain,dvalidation,dtest
            elif mode=='static':
                xtrain,xvalidation,xtest=strain,svalidation,stest
            else:
                xtrain=[dtrain,strain]
                xvalidation=[dvalidation,svalidation]
                xtest=[dtest,stest]

            early_stopping=tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',patience=3,min_delta=1e-4,restore_best_weights=True)
            history=model.fit(
                xtrain,weekly_y[train],validation_data=(xvalidation,weekly_y[validation]),
                epochs=30,batch_size=64,class_weight=class_weights,
                callbacks=[early_stopping],verbose=0)
            predictions[model_name][test]=model.predict(xtest,batch_size=256,verbose=0).ravel()
            training_log.append({'fold':fold,'model':model_name,'epochs':len(history.history['loss'])})
            print(f'Weekly fold {fold}: {model_name}; epochs={len(history.history["loss"])}')

    rows=[{'model':name,**calculate_metrics(weekly_y,p)} for name,p in predictions.items()]
    return pd.DataFrame(rows),predictions,pd.DataFrame(training_log)

RUN_WEEKLY=True  # Set to False to load the saved verified metrics without retraining.
if RUN_WEEKLY:
    weekly_results,weekly_predictions,weekly_training_log=run_weekly_models()
    weekly_results.to_csv(RESULTS/'weekly_ablation_metrics.csv',index=False)
    pd.DataFrame({'USUBJID':weekly_groups,'outcome':weekly_y,**weekly_predictions}).to_csv(RESULTS/'weekly_three_model_predictions.csv',index=False)
    weekly_training_log.to_csv(RESULTS/'weekly_three_model_training_log.csv',index=False)
else:
    weekly_results=pd.read_csv(RESULTS/'weekly_ablation_metrics.csv')

print(weekly_results.round(2).to_string(index=False))

column_names={
    'model':'Model','accuracy_pct':'Accuracy, %','precision_pct':'Precision, %',
    'recall_pct':'Recall, %','specificity_pct':'Specificity, %'}
daily_table=daily_results.rename(columns=column_names)[list(column_names.values())].round(2)
weekly_table=weekly_results.rename(columns=column_names)[list(column_names.values())].round(2)

print('Table 7. Performance comparison of dynamic-only, static-only, and integrated models for daily adherence prediction')
print(daily_table.to_string(index=False))
print('Table 8. Performance comparison of dynamic-only, static-only, and integrated models for weekly adherence prediction')
print(weekly_table.to_string(index=False))

daily_table.to_csv(RESULTS/'table7_daily_three_model_ablation.csv',index=False)
weekly_table.to_csv(RESULTS/'table8_weekly_three_model_ablation.csv',index=False)

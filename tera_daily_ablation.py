import os, random, json
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED=20260811
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
DATA='TERA Analysis/exploratory_analysis/tera_daily.csv'
df=pd.read_csv(DATA).dropna(subset=['USUBJID','withinrange']).reset_index(drop=True)
df['USUBJID']=df.USUBJID.astype(int)
y=df.withinrange.astype(int).to_numpy(); groups=df.USUBJID.to_numpy()
per_day=['is_Weekend','is_adherent','Morning','Afternoon','Evening','Night']
dyn_cols=[f't-{lag} {feat}' for lag in range(7,0,-1) for feat in per_day]
excluded={'USUBJID','withinrange','ADY','Morning','Afternoon','Evening','Night','is_adherent'}
static_cols=[c for c in df.columns if c not in excluded and c not in dyn_cols]
Xd=df[dyn_cols].to_numpy(float).reshape(-1,7,len(per_day))
Xs=df[static_cols].to_numpy(float)

def prep(train_idx, val_idx, test_idx):
    a=Xd[train_idx].reshape(len(train_idx),-1); b=Xd[val_idx].reshape(len(val_idx),-1); c=Xd[test_idx].reshape(len(test_idx),-1)
    impd=SimpleImputer(strategy='median').fit(a); a=impd.transform(a); b=impd.transform(b); c=impd.transform(c)
    scd=StandardScaler().fit(a); a=scd.transform(a); b=scd.transform(b); c=scd.transform(c)
    a=a.reshape(-1,7,len(per_day)); b=b.reshape(-1,7,len(per_day)); c=c.reshape(-1,7,len(per_day))
    imps=SimpleImputer(strategy='median').fit(Xs[train_idx]); sa=imps.transform(Xs[train_idx]); sb=imps.transform(Xs[val_idx]); sc=imps.transform(Xs[test_idx])
    scs=StandardScaler().fit(sa); sa=scs.transform(sa); sb=scs.transform(sb); sc=scs.transform(sc)
    return a,b,c,sa,sb,sc

def model(integrated, nstatic):
    di=tf.keras.Input((7,len(per_day)),name='dynamic')
    x=tf.keras.layers.LSTM(128,return_sequences=True)(di)
    x=tf.keras.layers.LSTM(128)(x)
    inputs=[di]
    if integrated:
        si=tf.keras.Input((nstatic,),name='static')
        s=tf.keras.layers.Dense(32,activation='relu')(si)
        x=tf.keras.layers.Concatenate()([x,s]); inputs.append(si)
    x=tf.keras.layers.Dropout(.25)(x)
    x=tf.keras.layers.Dense(32,activation='relu')(x)
    out=tf.keras.layers.Dense(1,activation='sigmoid')(x)
    m=tf.keras.Model(inputs,out)
    m.compile(tf.keras.optimizers.Adam(1e-4),loss='binary_crossentropy')
    return m

outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
probs={k:np.full(len(df),np.nan) for k in ['EDM-only','Integrated']}
fold_rows=[]
for fold,(tr_all,te) in enumerate(outer.split(np.zeros(len(y)),y,groups)):
    # Inner validation is also separated by participant.
    gss=GroupShuffleSplit(1,test_size=.20,random_state=SEED+fold)
    ti,vi=next(gss.split(tr_all,y[tr_all],groups[tr_all])); tr=tr_all[ti]; va=tr_all[vi]
    dtr,dva,dte,str_,sva,ste=prep(tr,va,te)
    n1=y[tr].sum(); n0=len(tr)-n1; cw={0:len(tr)/(2*n0),1:len(tr)/(2*n1)}
    for name,integrated in [('EDM-only',False),('Integrated',True)]:
        tf.keras.backend.clear_session(); tf.random.set_seed(SEED+fold)
        m=model(integrated,len(static_cols))
        xin=[dtr,str_] if integrated else dtr; xval=[dva,sva] if integrated else dva; xte=[dte,ste] if integrated else dte
        es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,min_delta=1e-4,restore_best_weights=True)
        h=m.fit(xin,y[tr],validation_data=(xval,y[va]),epochs=30,batch_size=64,class_weight=cw,callbacks=[es],verbose=0)
        probs[name][te]=m.predict(xte,batch_size=256,verbose=0).ravel()
        fold_rows.append({'fold':fold+1,'model':name,'epochs':len(h.history['loss'])})

def calc(p):
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy_pct':100*accuracy_score(y,pred),'precision_pct':100*precision_score(y,pred,zero_division=0),
            'recall_pct':100*recall_score(y,pred,zero_division=0),'specificity_pct':100*tn/(tn+fp)}
res={k:calc(v) for k,v in probs.items()}
pd.DataFrame([{'model':k,**v} for k,v in res.items()]).to_csv('results/daily_ablation_metrics.csv',index=False)
pd.DataFrame(fold_rows).to_csv('results/daily_ablation_folds.csv',index=False)
np.savez('results/daily_ablation_predictions.npz',y=y,groups=groups,**{k.replace('-','_'):v for k,v in probs.items()})
print(json.dumps({'participants':int(df.USUBJID.nunique()),'observations':len(df),'results':res,'epochs':fold_rows},indent=2))

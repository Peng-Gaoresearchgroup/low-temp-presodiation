import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")
import yaml,os,sys
import utils
from sklearn.metrics import accuracy_score, classification_report
current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from model.rfc import RFC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib,random,sys
import numpy as np
from sklearn.model_selection import KFold



def load_conf():
    with open("./conf/conf.yaml") as f:
        conf = utils.SafeDict(yaml.safe_load(f))
    return conf

conf=load_conf()
random.seed(conf.seed)
np.random.seed(conf.seed)


opt_rfc=joblib.load("./model/quick_test/rfc_model.pkl")
# opt_rfc=RFC()
n_estimators=opt_rfc.n_estimators
max_depth=opt_rfc.max_depth
cols_rm=opt_rfc.col_rm



def normalized_descriptors(descriptors):
    scaler = MinMaxScaler()
    normal_des = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)
    normal_des.apply(lambda x: round(x, 8))
    return normal_des

def make_k_flod_data(X,Y):
    k=10
    seed=42
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for i,(train_idx, val_idx) in enumerate(kf.split(X=X,y=Y)):
        folds.append((
            X.iloc[train_idx],
            Y.iloc[train_idx],
            X.iloc[val_idx],
            Y.iloc[val_idx]
        ))
    return folds



def train_rfc(train_X,train_Y,val_X,val_Y):
    r=RFC(X=train_X,Y=train_Y,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=0.3,
                    random_state=42,
                    min_samples_leaf=6,
                    min_samples_split=2,
                    oob_score=False)
    r.fit(train_X,train_Y)
    # y_train_pred=r.predict(train_X)
    # y_val_pred=r.predict(val_X)
    # return y_train_pred,y_val_pred
    return r

def k_fold_val(folds):
    records=[]
    for i,(train_X,trian_Y,val_X,val_Y) in enumerate(folds):
        model=train_rfc(train_X,trian_Y,val_X,val_Y)
        y_pred_train=model.predict(train_X)
        y_pred_val = model.predict(val_X)
        # train_accu=accuracy_score(y_true=trian_Y,y_pred=y_pred_train)
        # val_accu=accuracy_score(y_true=val_Y,y_pred=y_pred_val)
        # error[0].append(1-train_accu)
        # error[1].append(1-val_accu)
        # print(f"{error},")
        # print(f"-------------------{i}------------")
        report=classification_report(val_Y,y_pred_val,output_dict=True)
        records.append(report)
        # print(type(report))
        # print(report["1"]["precision"])
        # print(classification_report(trian_Y,y_pred_train))
    final_report=records[0]
    # del final_report["accuracy","macro avg","weighted avg"]
    for i in ["1","2"]:
        for j in ["precision","recall","f1-score","support"]:
            tmp=[k[i][j] for k in records]
            final_report[i][j]=f"{np.mean(tmp):.2f}+-{np.sqrt(np.var(tmp)):.2f}"

    print(final_report)
    return records

def categroy_al(al):
        if isinstance(al, (int, float)):
            if al<1:
                return 0
            elif 1<= al <2.5:
                return 1
            elif 2.5<= al <4:
                return 2
            elif 4<= al <5.5:
                return 3
            else:
                return 4
        else:
            return None





data=pd.read_csv("./data/AllProps_1400BzNSN.csv")

data=data[["SMILES","Ered"]]
# vs Li+/Li to vs Na+/Na
data["Ered"]=data["Ered"]-0.33
data["category"]=data["Ered"].apply(categroy_al)

# print(data.head(10))
if not os.path.exists("./data/1400BzNSN_descriptors.csv"):
    data=data.join(data['SMILES'].apply(lambda x: pd.Series(utils.smiles2descirptors(smiles=x,conf=conf))))
    X=data[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy]
    X.to_csv("./data/1400BzNSN_descriptors.csv",index=False)

X=pd.read_csv("./data/1400BzNSN_descriptors.csv")

if not os.path.exists("./data/1400BzNSN_X.csv"):
    X=normalized_descriptors(X)
    X=X.drop(columns=cols_rm)
    X.to_csv("./data/1400BzNSN_X.csv",index=False)

X=pd.read_csv("./data/1400BzNSN_X.csv")
        
Y=data["category"]

folds=make_k_flod_data(X,Y)
k_fold_val(folds)

final_model=train_rfc(X,Y,None,None)

#re scale
s_X=pd.read_csv("./data/1400BzNSN_descriptors.csv")
print("--------------------",len(s_X.columns))
scaler = MinMaxScaler()
scaler.fit(s_X)

scaler2 = MinMaxScaler()
print("--------------------",len(pd.read_csv("./data/data.csv")
[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy].columns))
scaler2.fit(pd.read_csv("./data/data.csv")[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy])
print("--------------------Sclae Range")
print(scaler.data_max_,scaler.data_min_)
print("--------------------Sclae Range")
print(scaler2.data_max_,scaler2.data_min_)


candidate_X=joblib.load("./model/quick_test/candidate_X.pkl")
print("--------------------",len(candidate_X.columns))
print((candidate_X.columns==s_X.columns))

print(candidate_X.head(5))

candidate_X=scaler2.inverse_transform(candidate_X)
candidate_X=scaler.transform(candidate_X)
candidate_X=pd.DataFrame(candidate_X,columns=s_X.columns)
# candidata_X=candidata_X.drop(columns=cols_rm)

print(candidate_X.columns)
candidate_X=candidate_X.drop(columns=cols_rm)
if not os.path.exists("./outputs/train_BzNSN/candidate_X_rescaled.pkl"):
    joblib.dump(candidate_X,"./outputs/train_BzNSN/candidate_X_rescaled.pkl")
print(candidate_X.head(5))
candidata_Y_pred=final_model.predict(candidate_X)
prob_matrix=final_model.predict_proba(candidate_X)
if not os.path.exists("./outputs/train_BzNSN/candidate_pred_prob.csv"):
    np.savetxt("./outputs/train_BzNSN/candidate_pred_prob.csv",prob_matrix)
candidate_smiles=pd.read_csv("./data/data.csv")[["canonicalsmiles",'comment']][216:]
candidate_smiles["y_pred"]=candidata_Y_pred
print(len(candidate_smiles))
candidate_smiles.to_csv("./outputs/train_BzNSN/candidate_pred.csv")


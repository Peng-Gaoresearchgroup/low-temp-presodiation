import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from model.rfc import RFC
from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
k=10
seed=42
def load_model():
    model=joblib.load('./model/quick_test/rfc_model.pkl')
    return model
optimal_rfc=load_model()
n_estimators=optimal_rfc.n_estimators
max_depth=optimal_rfc.max_depth

def load_data():
    known_X=joblib.load('./model/quick_test/rfc_X.pkl')
    known_Y=joblib.load('./model/quick_test/rfc_Y.pkl')
    candidate_X=joblib.load('./model/quick_test/candidate_X.pkl')
    # print(type(known_X))
    # print(known_X[:10],known_Y[:10])
    return known_X,known_Y,candidate_X


def make_k_flod_data(X,Y):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    # print(kf.split(X,Y))
    # for tran_idx,val_idx()
    for i,(train_idx, val_idx) in enumerate(kf.split(X=X,y=Y)):
        # print(f"K={i}")
        # print(train_idx)
        # print(val_idx)
        # break
        folds.append((
            X.iloc[train_idx],
            Y.iloc[train_idx],
            X.iloc[val_idx],
            Y.iloc[val_idx]
        ))
    return folds


def train_rfc(train_X,train_Y,val_X,val_Y,alpha):
    r=RFC(X=train_X,Y=train_Y,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=0.3,
                    random_state=42,
                    min_samples_leaf=6,
                    min_samples_split=2,
                    oob_score=False)
    r._get_external_splited_dataset(train_X,train_Y,val_X,val_Y)
    r._lasoo(alpha=alpha,threshold=0.1) # Lasso regularization, filter low correlation features
    r._fit_by_train_dataset()
    r._cal_accuracy_train()
    r._cal_accuracy_test()
    # print()
    return r


def k_fold_val(folds):
    error=[[],[]]
    for (train_X,trian_Y,val_X,val_Y) in folds:
        model=train_rfc(train_X,trian_Y,val_X,val_Y,0.001)
        y_pred_train=model._predict_after_lasso(train_X)
        y_pred_val = model._predict_after_lasso(val_X)
        train_accu=accuracy_score(y_true=trian_Y,y_pred=y_pred_train)
        val_accu=accuracy_score(y_true=val_Y,y_pred=y_pred_val)
        error[0].append(1-train_accu)
        error[1].append(1-val_accu)
        # print(f"{error},")
    error=np.matrix_transpose(error)
    np.savetxt("./outputs/model_val/k_error.txt",np.array(error))
    return error

def k_fold_val_report(folds):
    print("------------K Fold validation Report-----------")
    records=[]
    for i,(train_X,trian_Y,val_X,val_Y) in enumerate(folds):
        model=train_rfc(train_X,trian_Y,val_X,val_Y,0.001)
        y_pred_train=model._predict_after_lasso(train_X)
        y_pred_val = model._predict_after_lasso(val_X)
        # train_accu=accuracy_score(y_true=trian_Y,y_pred=y_pred_train)
        # val_accu=accuracy_score(y_true=val_Y,y_pred=y_pred_val)
        # error[0].append(1-train_accu)
        # error[1].append(1-val_accu)
        # print(f"{error},")
        # print(f"-------------------{i}------------")
        report=classification_report(val_Y,y_pred_val,output_dict=True)
        del report["accuracy"]
        del report["macro avg"]
        del report["weighted avg"]
        
        for i in ['0.0',"1.0","2.0","3.0","4.0"]:
            if i not in report.keys():
                report[i]={"precision":999,
                           "recall":999,
                           "f1-score":999,
                           "support":999}
        records.append(report)
        # print(classification_report(trian_Y,y_pred_train))
    final_report=records[0]

    # print(records)
    for i in ['0.0',"1.0","2.0","3.0","4.0"]:
        for j in ["precision","recall","f1-score","support"]:
            tmp=[k[i][j] for k in records if k[i][j]!=999]
            final_report[i][j]=f"{np.mean(tmp):.2f}+-{np.sqrt(np.var(tmp)):.2f}"

    print(final_report)
    print("------------K Fold validation Report-----------")

def k_folds_performance_curve(folds):
    y_true_list=[]
    y_score_list=[]
    for j,(train_X,trian_Y,val_X,val_Y) in enumerate(folds):
        model=train_rfc(train_X,trian_Y,val_X,val_Y,0.001)
        target_class = 2
        y_true_binary = (val_Y == target_class).astype(int)
        y_score = model._predict_proba_after_lasso(val_X)[:, target_class]
        for i,y in enumerate(y_true_binary):
            y_true_list.append(y)
            y_score_list.append(y_score[i])

        fpr, tpr, _ = roc_curve(y_true_list, y_score_list)
        precision, recall, _ = precision_recall_curve(y_true_list, y_score_list)
        roc=np.column_stack([fpr, tpr])
        pr=np.column_stack([recall, precision])
        np.savetxt(f"./outputs/model_val/roc_{j}.txt",roc)
        np.savetxt(f"./outputs/model_val/pr_{j}.txt",pr)


def optimal_model_pred_matrix(model:RFC):
    y_true_train=model.Y_train
    y_train_score=model.predict_proba(model.X_train)
    df=pd.DataFrame(y_train_score,index=model.X_train.index, columns=[0,1,2,3])
    df["y_true_train"]=y_true_train
    df=df.sort_values(by="y_true_train")
    df.to_csv("./outputs/model_val/optimal_rfc_prob_train.csv",index=True)

    y_true_val=model.Y_test
    y_val_score=model.predict_proba(model.X_test)
    df2=pd.DataFrame(y_val_score,index=model.X_test.index, columns=[0,1,2,3])
    df2["y_true_val"]=y_true_val
    df2=df2.sort_values(by="y_true_val")
    df2.to_csv("./outputs/model_val/optimal_rfc_prob_val.csv",index=True)
    return df,df2

def optimal_model_prob_mean_var(df,df2):
    #train
    print("train")
    print("groud truth,smapel num,pred_0,pred_1,pred_2,pred_3")
    for i in range(4):
        df_tmp=df[df["y_true_train"]==i]
        print(i,len(df_tmp),end=" ")
        for j in range(4):
            print("({:.3f}±{:.3f}),".format(np.mean(df_tmp[j]),np.sqrt(np.var(df_tmp[j]))),end="")
        print("")
    #test
    print("test")
    print("groud truth,smapel num,pred_0,pred_1,pred_2,pred_3")
    for i in range(4):
        df_tmp=df2[df2["y_true_val"]==i]
        print(i,len(df_tmp),end=" ")
        for j in range(4):
            print("({:.3f} ± {:.3f}),".format(np.mean(df_tmp[j]),np.sqrt(np.var(df_tmp[j]))),end="")
        print("")

def optimal_model_entropy_and_confidence(df,df2):
    eps=1e-12
    df["entropy"]=-(df[0]*np.log(df[0]+eps)+\
                    df[1]*np.log(df[1]+eps)+\
                    df[2]*np.log(df[2]+eps)+\
                    df[3]*np.log(df[3]+eps))
    entopy_train=(np.mean(df["entropy"]),np.var(df["entropy"]))
    
    df2["entropy"]=-(df2[0]*np.log(df2[0]+eps)+\
                    df2[1]*np.log(df2[1]+eps)+\
                    df2[2]*np.log(df2[2]+eps)+\
                    df2[3]*np.log(df2[3]+eps))
    
    entopy_test=(np.mean(df2["entropy"]),np.var(df2["entropy"]))
    

    df["confidence"] = df[[0, 1, 2, 3]].max(axis=1)
    df2["confidence"] = df2[[0, 1, 2, 3]].max(axis=1)
    confidence_train=(np.mean(df["confidence"]),np.var(df["confidence"]))
    confidence_val=(np.mean(df2["confidence"]),np.var(df2["confidence"]))
    
    
    print(entopy_train)
    print(entopy_test)
    print(confidence_train)
    print(confidence_val)
    df.to_csv("./outputs/model_val/optimal_rfc_entropy_confidence_train.csv",index=True)
    df2.to_csv("./outputs/model_val/optimal_rfc_entropy_confidence_test.csv",index=True)

def optimal_model_infomation(opt_model:RFC):
    x=opt_model.X_test
    y=opt_model.Y_test
    y_pred=opt_model.predict(x)
    # acc=accuracy_score(y,y_pred)
    print(classification_report(y,y_pred))

def performance_without_lasso(X:pd.DataFrame,Y:pd.DataFrame,opt_model:RFC):
    print("Performacen without lasso")
    X_train=X.loc[opt_model.X_train.index]
    Y_train=Y.loc[opt_model.Y_train.index]
    X_test=X.loc[opt_model.X_test.index]
    Y_test=Y.loc[opt_model.Y_test.index]
    # print(len(X_train),len(opt_model.X_train))
    n_estimators=opt_model.n_estimators
    max_depth=opt_model.max_depth
    rfc=RandomForestClassifier(n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=0.3,
                    random_state=42,
                    min_samples_leaf=6,
                    min_samples_split=2,
                    oob_score=False)
    rfc.fit(X_train,Y_train)
    y_pred=rfc.predict(X_test)
    print(classification_report(Y_test,y_pred))


X,Y,candidate_X=load_data()
folds=make_k_flod_data(X,Y)
print(k_fold_val(folds))

# k_folds_performance_curve(folds)
df_train,df_test=optimal_model_pred_matrix(optimal_rfc)
# optimal_model_entropy_and_confidence(df_train,df_test)
optimal_model_prob_mean_var(df_train,df_test)
optimal_model_infomation(optimal_rfc)
performance_without_lasso(X,Y,optimal_rfc)

print(np.mean([i[0] for i in k_fold_val(folds)]),np.sqrt(np.var([i[0] for i in k_fold_val(folds)])))
print(np.mean([i[1] for i in k_fold_val(folds)]),np.sqrt(np.var([i[1] for i in k_fold_val(folds)])))
k_fold_val_report(folds)
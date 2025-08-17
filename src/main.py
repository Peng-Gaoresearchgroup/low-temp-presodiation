import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")
import yaml,os,sys
import utils

current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from model import rfc,pareto
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib,random,sys
import numpy as np

# import inspect

def log(func):
    import time
    def wrapper(*args, **kwargs):
        t=time.strftime("%H:%M:%S", time.localtime())
        print(f"[INFO {t}] 调用 {func.__name__}()")
        if func.__name__=='to_csv':
            print(f'               {args[1]} 已保存')

        return func(*args, **kwargs)
    return wrapper

pd.DataFrame.to_csv=log(pd.DataFrame.to_csv)

@log
def load_conf():
    with open("./conf/conf.yaml") as f:
        conf = utils.SafeDict(yaml.safe_load(f))
    return conf

@log
def wash_data():
    import subprocess
    subprocess.run("python wash.py", shell=True, check=True)

@log
def normalized_descriptors(descriptors):
    
    scaler = MinMaxScaler()
    normal_des = pd.DataFrame(scaler.fit_transform(descriptors), columns=descriptors.columns)
    normal_des.apply(lambda x: round(x, 8))
    return normal_des

@log
def prepare_rank_data(df:pd.DataFrame):

    df['solv_energy']=df['product'].apply(utils.get_solv)
    # df['capcacity']=
    df[['sascore','scscore','spatial']]=df['canonicalsmiles'].apply(lambda x :pd.Series(utils.get_scscore(x)))
    return df

@log
def train_rfc(rfc_X,rfc_Y,n_estimators,max_depth,alpha):
    r=rfc.RFC(X=rfc_X,Y=rfc_Y,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=0.3,
                    random_state=42,
                    min_samples_leaf=6,
                    min_samples_split=2,
                    oob_score=False)
    r._split_dataset(test_size=0.3)
    r._lasoo(alpha=alpha,threshold=0.1) # Lasso regularization, filter low correlation features
    r._fit_by_train_dataset()
    r._cal_accuracy_train()
    r._cal_accuracy_test()
    return r

@log
def search_hyperpara(rfc_X,rfc_Y,n_estimators_range:list,max_depth_range:list):
    d={'n_estimators':[],'max_depth':[],'accuracy_train':[],'accuracy_test':[]}
    # d={'n_estimators':[]}
    # for j in range(max_depth_range[0],max_depth_range[1]):
    #     d.update({j:[]})
    for i in range(n_estimators_range[0],n_estimators_range[1]):
        # d['n_estimators'].append(i)
        for j in range(max_depth_range[0],max_depth_range[1]):
            r=train_rfc(rfc_X,rfc_Y,n_estimators=i,max_depth=j,alpha=0.001)
            accuracy_train,accuracy_test=r.accuracy_train,r.accuracy_test
            # d[j].append(accuracy_train)
            d['n_estimators'].append(i)
            d['max_depth'].append(j)
            d['accuracy_train'].append(accuracy_train)
            
            d['accuracy_test'].append(accuracy_test)
            
            print(f'Searching: n_estimators = {i}, max_depth = {j}')
    return pd.DataFrame(d)

@log
def find_optimal_hyperpara(df:pd.DataFrame):
    max_index = df['accuracy_test'].idxmax()
    n_estimators=df.loc[max_index,'n_estimators']
    max_depth=df.loc[max_index,'max_depth']
    return n_estimators,max_depth

@log
def search_alpha_lasso(rfc_X,rfc_Y):
    # r=train_rfc(rfc_X,rfc_Y,n_estimators=14,max_depth=4)
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import train_test_split
    r=rfc.RFC(X=rfc_X,Y=rfc_Y,
                    n_estimators=14,
                    max_depth=4,
                    max_features=0.3,
                    random_state=42,
                    min_samples_leaf=6,
                    min_samples_split=2,
                    oob_score=False)
    r._split_dataset(test_size=0.3)
    # print(len(r.X_train.columns),r.X_train.columns)

    d={'alpha':[]}
    for i in range(43):
        d.update({f'Feature{i}':[]})
    for i in [0.00001,0.0001,0.001,0.01,0.1,1]:
        r._lasoo(i,threshold=0)
        q=list(r.lasso_result)
        d['alpha'].append(i)
        for i in range(len(q)):
            d[f'Feature{i}'].append(q[i])
    return pd.DataFrame(d)

@log
def get_rank_data(df):
    df[['sascore','scscore','spacial_score']]=df['canonicalsmiles'].apply(lambda x: pd.Series(utils.get_scscore(x)))
    df['product_solv_energy']=df['product'].apply(utils.find_product_solvenergy)
    df['product_mp']=df['product'].apply(utils.find_product_mp)
    df['anode_limit']=df['canonicalsmiles'].apply(utils.find_anode_limt)
    return df

@log
def score(df):
    for i in ['product_solv_energy','sascore','scscore','spacial_score','product_mp','capacity','anode_limit']:
        # df[f'nor_{i}']=df[i].apply(lambda x : (x- x.min())/(x.max()-x.min()))
        df[f'nor_{i}'] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
    df['score']=df['nor_capacity']-df['nor_product_solv_energy']-df['nor_product_mp']-df['anode_limit']-df['nor_sascore']-df['nor_scscore']-df['nor_spacial_score']
    return df

@log
def quick_test():
    # load data and model
    rfc_X=joblib.load('./model/quick_test/rfc_X.pkl')
    rfc_Y=joblib.load('./model/quick_test/rfc_Y.pkl')
    candidate_X=joblib.load('./model/quick_test/candidate_X.pkl')
    r=joblib.load('./model/quick_test/rfc_model.pkl')
    print(f'Optimal RFC:\nn_estimators: {r.n_estimators}\n max_depth: {r.max_depth}\nlasso_result: {r.lasso_result}\nSelectted columns: {r.X_train.columns}')

    
    y=r._predict_after_lasso(X=candidate_X)
    print(f'Predict: {y}')

    recommed=joblib.load('./model/quick_test/recommend_molecules.pkl')
    print(f'Recommend molecule:{recommed}')
    
    front=joblib.load('./model/quick_test/pareto_front.pkl')
    print(f'Pareto front:{front}')


@log
def main():
    # load data and conf
    conf=load_conf()
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    data=pd.read_csv(conf.data_path)

    des=data[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy]
    features=normalized_descriptors(des)

    # RFC
    # Split kown data and candidate data, prepare for input into RFC
    split_line=216 
    rfc_X=features[0:split_line]
    rfc_Y=data['comment'][0:split_line]
    candidate_X=features[split_line:]
    joblib.dump(rfc_X,'./model/quick_test/rfc_X.pkl')
    joblib.dump(rfc_Y,'./model/quick_test/rfc_Y.pkl')
    joblib.dump(candidate_X,'./model/quick_test/candidate_X.pkl')
    
    # Search hyperparameters: n_estimators and max_depth
    hyperpara_df=search_hyperpara(rfc_X,rfc_Y,n_estimators_range=[1,100],max_depth_range=[1,10])
    hyperpara_df.to_csv('./outputs/rfc_hyperpara_search.csv',index=False)
    
    # Find optimal hyperparameters
    best_n_estimators,best_max_depth=find_optimal_hyperpara(hyperpara_df)
    
    # Instantiate the optimal class
    r=train_rfc(rfc_X,rfc_Y,n_estimators=best_n_estimators,max_depth=best_max_depth,alpha=0.001)
    joblib.dump(r,'./model/quick_test/rfc_model.pkl')
    print(f'Optimal RFC:\nn_estimators: {best_n_estimators}\n max_depth: {best_max_depth}\nlasso_result: {r.lasso_result}\nSelectted columns: {r.X_train.columns}')

    # Predict candidate molecules
    y=r._predict_after_lasso(X=candidate_X)
    data2=data[split_line:].copy()
    data2['Y_predict']=y
    data2=data2[['idx','canonicalsmiles','type','R','product','capacity','Y_predict']]
    data2.to_csv('./outputs/rfc_predict.csv',index=False)
    print(f'Recommend num of candidate mols:{len(data2[data2["Y_predict"]==2])}')

    recommed=data2[data2["Y_predict"]==2]
    recommed.to_csv('./outputs/recommend_mols.csv',index=False)

    # Rank

    recommed=pd.read_csv('./outputs/recommend_mols_with_rankdata.csv')
    recommed=recommed[(recommed['anode_limit']<=4.0)&(recommed['anode_limit']>=2.5)]
    recommed=recommed[recommed['product_mp']<=999]
    recommed=score(recommed)
    joblib.dump(recommed,'./model/quick_test/recommend_molecules.pkl')

    # Pareto
    p=pareto.Pareto(df=recommed,
                    opt_target={
                        'product_solv_energy':'min',
                                'sascore':'min',
                                'scscore':'min',
                                'spacial_score':'min',
                                'product_mp':'min',
                                # 'capacity':'max',
                                # 'anode_limit':'min'
                                })
    front=p.pareto_front()
    p=pareto.Pareto(df=front,
            opt_target={
                # 'product_solv_energy':'min',
                #         'sascore':'min',
                #         'scscore':'min',
                #         'spacial_score':'min',
                #         'product_mp':'min',
                        'capacity':'max',
                        'anode_limit':'min'
                        })
    
    front=p.pareto_front()
    joblib.dump(front,'./model/quick_test/pareto_front.pkl')
    front=front.sort_values(by='score',ascending=False)
    front.to_csv('./outputs/pareto_fornt.csv',index=False)
            
if __name__=='__main__':
    test=int([i for i in sys.argv if i.startswith('test=')][0].split('=')[1])
    if test==1:
        pass
        quick_test()
    elif test==0:
        main()

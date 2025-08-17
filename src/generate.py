import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import random,yaml
import numpy as np
import utils

def log(func):
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__} 函数，参数：{args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper
def apply_log_decorator_to_all_functions():
    for name, obj in globals().items():
        if callable(obj):
            globals()[name] = log(obj)
apply_log_decorator_to_all_functions()

def activate_center():
    d={'S-':'[*:1][S-]',
       'SO2-':'[*:1]S(=O)[O-]',
       'SO3-':'[*:1]S(=O)(=O)[O-]',
       'N-':'[*:1][N-][*:1]',
       'NO2--':'[*:1][N]([O-])[O-]',
       'NN--':'[*:1][N-][N-][*:1]',
       'O-':'[*:1][O-]',
       'COO-':'[*:1]C(=O)[O-]',
       'COOCOO-':'[*:1]OC(=O)C(=O)[O-]',
       'P-':'[P-]([*:1])([*:1])([*:1])([*:1])([*:1])([*:1])',
       'B-':'[B-]([*:1])([*:1])([*:1])([*:1])',
       'PFO2-':'[*:1][P](F)(=O)[O-]',
       'NO--':'[*:1][N-][O-]',
        }
    return d

def R_groups():
    l=['C',                                                        # 甲基
       'C(F)(F)F',                                                 # 三氟甲基
       'CC',                                                       # 乙基
       'C(F)(F)C(F)(F)F',                                          # 全氟乙基
       'CCC',                                                      # 丙基
       'C(F)(F)C(F)(F)C(F)(F)F',                                   # 全氟丙基
       'COC',                                                      # 甲醚甲
       'C(F)(F)OC(F)(F)F',                                         # 全氟甲醚甲
       'CCCC',                                                     # 丁基
       'C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',                            # 全氟丁基
       'COCC',                                                     # 甲醚乙
       'C(F)(F)OC(F)(F)C(F)(F)F',                                  # 全氟甲醚乙
       'CCOC',                                                     # 乙醚甲
       'C(F)(F)C(F)(F)OC(F)(F)F',                                   # 全氟乙醚甲
       'CCCCC',                                                    # 戊基
       'C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',                     # 全氟戊基
       'COCCC',                                                    # 甲醚丙
       'C(F)(F)OC(F)(F)C(F)(F)C(F)(F)F',                           # 全氟甲醚丙
       'CCOCC',                                                    # 乙醚乙
       'C(F)(F)C(F)(F)OC(F)(F)C(F)(F)F',                           # 全氟乙醚乙
       'CCCOC',                                                    # 丙醚甲
       'C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F',                           # 全氟丙醚甲
       ]                      
    return l

def combine(center,group,site='[*:1]'):
    c=Chem.MolFromSmiles(center)
    g=Chem.MolFromSmiles(group)
    product = AllChem.ReplaceSubstructs(c,Chem.MolFromSmiles(site), g, replaceAll=True)
    return Chem.MolToSmiles(product[0])

def combine_product(tp,group,site='[*:1]'):
    c=None
    if tp=='S-':
        c=Chem.MolFromSmiles('[*:1]SS[*:1]')
    elif tp=='NN--':
        c=Chem.MolFromSmiles('[*:1][N]=[N][*:1]')
    elif tp=='N-':
        c=Chem.MolFromSmiles('[*:1][N]([*:1])[N]([*:1])[*:1]')
    elif tp=='NO2--':
        c=Chem.MolFromSmiles('[*:1][N](=[O])=[O]')
    elif tp=='O-':
        c=Chem.MolFromSmiles('[*:1]OO[*:1]')
    elif tp=='NO--':
        c=Chem.MolFromSmiles('[*:1][N]=[O]')
    else:
        c=Chem.MolFromSmiles('[*:1][*:1]')
    g=Chem.MolFromSmiles(group)
    product = AllChem.ReplaceSubstructs(c,Chem.MolFromSmiles(site), g, replaceAll=True)
    return Chem.MolToSmiles(product[0])

def load_conf():
    with open("./conf/conf.yaml") as f:
        conf = utils.SafeDict(yaml.safe_load(f))
    return conf

def calculate_descriptors(conf,mols):
    mols=mols.join(mols['canonicalsmiles'].apply(lambda x: pd.Series(utils.smiles2descirptors(smiles=x,conf=conf))))
    return mols


def al_comment(al):
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

def main():
    #读取240分子
    df_train=pd.read_csv('./data/labeled.csv')
    df_train['anode_limit']=df_train['Voltage'] 
    df_train['canonicalsmiles']=df_train['smiles'].apply(utils.Na2ion)
    df_train=df_train[['canonicalsmiles','anode_limit']]
    df_train[['type','R','product']]=None
    # df_train['anode_comment1']=df_train['anode_limit'].apply(al_comment_big)
    df_train['comment']=df_train['anode_limit'].apply(al_comment)

    # 生成新分子库
    d={'canonicalsmiles':[],'type':[],'R':[],'product':[]}
    for tp,center in activate_center().items():
        for group in R_groups():
            print(f'Processing:{center}+{group}')
            smiles=combine(center,group,'[*:1]')+'.[Na+]'    
            pro=combine_product(tp=tp,group=group)
            d['canonicalsmiles'].append(smiles)
            d['type'].append(tp)
            d['R'].append(group)
            d['product'].append(pro)
    df_tmp=pd.DataFrame(d)
    df_tmp[['anode_limit','comment']]=None
    
    # 合并分子库
    # df_tmp=df_tmp[df_train.columns]
    df=pd.concat([df_train,df_tmp],ignore_index=True)
    # print(df)

    # 标注电压
    # df['anode_comment']=df['anode_limit'].apply(al_comment)
    # 计算分子描述符
    conf=load_conf()
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    df[['molwt', 'capacity']] = df['canonicalsmiles'].apply(lambda smiles: pd.Series(utils.get_molwt_capacity(smiles)))
    # print(df)
    df=calculate_descriptors(conf=conf,mols=df)
    df=df.dropna(subset=conf.hc._get+conf.hc._2d+conf.hc._3d+conf.hc._diy)
    df=df.reset_index(drop=True)
    df['idx']=df.index
    print('len(df): ',len(df))
    
    df.to_csv('./data/data.csv',index=False)

    # 获取产物
    df_pro=df.copy()
    # df_pro['prodcut']=product
    # df_pro=calculate_descriptors(conf=conf,mols=df_pro)
    df_pro=df_pro.dropna(subset=['R'])
    # df_pro=df_pro[['canonicalsmiles','prodcut']]
    df_pro.to_csv('./data/product.csv',index=False)


if __name__=='__main__':
    main()
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
import pandas as pd
from typing import TypedDict
import re,time,json,joblib
import requests as rq
import numpy as np
#-----------------------------------------------------
# Modifying a dictionary to make indexing easier, dict['A']['B'] == c ----> dict.A.B == c
#-----------------------------------------------------

class SafeDict(dict):
    def __getattr__(self, name):
        value = self.get(name, None) 
        if isinstance(value, dict):
            return SafeDict(value)
        elif isinstance(value, list):
            return [SafeDict(item) if isinstance(item, dict) else item for item in value]
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __dir__(self):
        return super().__dir__() + list(self.keys())

def nowtime():
    import time
    x=f'{time.localtime().tm_year-2000}-{time.localtime().tm_mon}-{time.localtime().tm_mday}-{time.localtime().tm_hour}-{time.localtime().tm_min}-{time.localtime().tm_sec}'
    return x

#-----------------------------------------------------
# Data process and statistics
#-----------------------------------------------------
def get_valid_smiles(df):
    li=df['canonicalsmiles'].to_list()
    print(len(li))
    new=[]
    for i in li:
        if Chem.MolFromSmiles(i) != None:
            new.append(Chem.MolToSmiles(Chem.MolFromSmiles(i)))
    print(len(new))
    return pd.DataFrame({'canonicalsmiles':new})

def get_all_elements(df):
    li=df['canonicalsmiles'].to_list()
    element=set()
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            for atom in mol.GetAtoms():
                element.add(atom.GetSymbol())
    return element

def get_elements_distribution(df,elements):
    li=df['canonicalsmiles'].to_list()
    count = {element: 0 for element in elements}
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            atoms=[i.GetSymbol() for i in mol.GetAtoms()]
            for element in elements:
                if element in atoms:
                    count[element] += 1
    sorted_count=sorted(count.items(), key=lambda x: x[1], reverse=False)
    return sorted_count

def get_molWt_distribution(df):
    li=df['canonicalsmiles'].to_list()
    molWt=[]
    for smiles in li:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol:
            molWt.append(Descriptors.MolWt(mol))
    return molWt

#-----------------------------------------------------
# Smiles process
#-----------------------------------------------------
def Na2ion(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        editable_mol = Chem.RWMol(mol)

    # 查找所有 Na 原子
        na_atoms = [atom for atom in editable_mol.GetAtoms() if atom.GetSymbol() == 'Na']
        
        for na_atom in na_atoms:
            # 查找与 Na 原子相连的所有邻居原子
            neighbors_to_modify = []
            for neighbor in na_atom.GetNeighbors():
                # 将邻居原子的电荷设置为 -1
                editable_mol.GetAtomWithIdx(neighbor.GetIdx()).SetFormalCharge(-1)
                neighbors_to_modify.append(neighbor.GetIdx())
            
            # 设置 Na 原子的电荷为 +1
            editable_mol.GetAtomWithIdx(na_atom.GetIdx()).SetFormalCharge(1)
            
            # 删除 Na 原子及其连接的共价键
            for neighbor_idx in neighbors_to_modify:
                editable_mol.RemoveBond(na_atom.GetIdx(), neighbor_idx)
            
            # editable_mol.RemoveAtom(na_atom.GetIdx())
        
        # 转换回 SMILES 字符串
        modified_smiles = Chem.MolToSmiles(editable_mol)
    except:
        if 'B' in smiles:
            smiles=smiles.replace('[Na]','.[Na+]')
            smiles=smiles.replace('B','[B-]')
            smiles=smiles.replace('[[','[')
            smiles=smiles.replace(']]',']')
            if '(.[Na+])' in smiles:
                l=smiles.split('(.[Na+])')
                modified_smiles=''.join(l)+'.[Na+]'
            else:
                modified_smiles=smiles
        elif 'P' in smiles:
            smiles=smiles.replace('[Na]','.[Na+]')
            smiles=smiles.replace('P','[P-]')
            smiles=smiles.replace('[[','[')
            smiles=smiles.replace(']]',']')
            if '(.[Na+])' in smiles:
                l=smiles.split('(.[Na+])')
                modified_smiles=''.join(l)+'.[Na+]'
            else:
                modified_smiles=smiles
    return modified_smiles

def get_molwt_capacity(smiles):
    try:
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        molwt = Descriptors.MolWt(mol)
        if '[Na]' in smiles:
            Na_num = smiles.count('[Na]')
        else:
            Na_num = smiles.count('-')
        capacity=Na_num*96485/(3.6*molwt)
        return molwt,capacity
    except:
        return None,None


def smiles2descirptors(smiles,conf): #desdic is a dic containing what descriptors we choose,{_get:['a','b'],_2d:['c']......}, see conf.yaml.
    '''Input a SMILES string, return its descriptors' pd.series calculated by rdkit'''
    seed=conf.seed
    desdic=conf.descriptors
    print(f'Processing {smiles}')    
    mol=Chem.MolFromSmiles(smiles) 
    if mol is None:
        raise ValueError("SMILES Error")
    try:
        # Get 2D descriptors
        des={f'{i}': getattr(mol, f'Get{i}')() for i in desdic._get}
        des.update({f'{i}': getattr(Descriptors, f'{i}')(mol) for i in desdic._2d})
        # 3D Embed
        AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=seed)
        AllChem.MMFFOptimizeMolecule(mol)
        mol=Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = seed
        AllChem.EmbedMolecule(mol, params)
        # Get 3D descriptors
        des.update({f'{i}': getattr(Descriptors3D, f'{i}')(mol) for i in desdic._3d})
        #Get Diy descriptors
        def SP2ratio(mol):
            total_c=0
            sp2=0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum()==6:
                    total_c+=1
                    hybridization=atom.GetHybridization()
                    if hybridization==Chem.HybridizationType.SP2:
                        sp2+=1
            sp2_frac=sp2/total_c
            return sp2_frac
        des.update({desdic._diy[0]:SP2ratio(mol)})

        def find_negatively_charged_atoms(mol):
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=seed)
                AllChem.MMFFOptimizeMolecule(mol)
                mol=Chem.AddHs(mol)
                params = AllChem.ETKDG()
                params.randomSeed = seed
                AllChem.EmbedMolecule(mol, params)
                Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
                ewg=['F','H','B','O']
                edg=['S','C','N','P','Si']
                # edg=['S','O','N']
                edg_num=0
                ewg_num=0
                # AllChem.ComputeGasteigerCharges(mol)  # 计算Gasteiger部分电荷

                charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
                # charges = [atom.GetDoubleProp('molAtomMapNumber') for atom in mol.GetAtoms()]
                # max_charge = max(charges)
                min_charge = min(charges)
                min_charge_atom = None
                
                for atom in mol.GetAtoms():
                    if atom.GetDoubleProp('_GasteigerCharge') == min_charge:
                        # min_charge_atom = atom.GetSymbol()
                        min_charge_atom = atom
                        break
                # print(min_charge_atom.GetSymbol())
                n=[neighbor for neighbor in min_charge_atom.GetNeighbors()]
                n_charge=[i.GetDoubleProp('_GasteigerCharge') for i in n]
                n_min_charge=min(n_charge)

                for i in n:
                    # print(i.GetSymbol())
                    if i.GetSymbol() in edg:
                            edg_num+=1
                    elif i.GetSymbol() in ewg:
                            ewg_num+=1
                    # for j in [neighbor.GetSymbol() for neighbor in i.GetNeighbors()]:
                    #     # print(j)
                    #     if j in edg:
                    #         edg_num+=1
                    #     elif j in ewg:
                    #         ewg_num+=1
                return min_charge_atom.GetAtomicNum(),edg_num/len(n),n_min_charge
            except:
                return None,None,None
        a,b,c=find_negatively_charged_atoms(mol)
        des.update({desdic._diy[1]:a})
        des.update({desdic._diy[2]:b})
        des.update({desdic._diy[3]:c})

        def element_ratio(mol):
            elements=['C','H','O','N','F','S','B','Si','Se','P']
            atoms=[i.GetSymbol() for i in mol.GetAtoms()]
            ratio=1/len(atoms)
            d={}
            for atom in elements:
                d.update({atom:0})
            for atom in atoms:
                if atom in elements:
                    d[atom]+=ratio

            return d['C'],d['H'],d['O'],d['N'],d['F'],d['S'],d['B'],d['Si'],d['Se'],d['P']
        c,h,o,n,f,s,b,si,se,p=element_ratio(mol)
        des.update({desdic._diy[4]:c})
        des.update({desdic._diy[5]:h})
        des.update({desdic._diy[6]:o})
        des.update({desdic._diy[7]:n})
        des.update({desdic._diy[8]:f})
        des.update({desdic._diy[9]:s})
        des.update({desdic._diy[10]:b})
        des.update({desdic._diy[11]:si})
        des.update({desdic._diy[12]:se})
        des.update({desdic._diy[13]:p})

            
        print(f'{smiles} Processing done')
        return des

    except:
        print(f'{smiles} Error')
        des={f'{i}': None for i in desdic._get}
        des.update({f'{i}': None for i in desdic._2d})
        des.update({f'{i}': None for i in desdic._3d})
        des.update({f'{i}': None for i in desdic._diy})
        return des

def extract_xy(string):
    pattern = r'\[\s?([\d\.\-eE]+)\s+([\d\.\-eE]+)\s?\s?\s?\]'
    match = re.match(pattern, string)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return x, y
    else:
        # raise ValueError(f"无法从字符串中提取 x 和 y: {string}")
        return None,None

#-----------------------------------------------------
# Molecule Ranking
#-----------------------------------------------------

def get_solv(product):
    idx_product=joblib.load('./DFT/product_index.pkl')
    idx=[k for k, v in idx_product.items() if v == product][0]
    # idx=idx_product.get()
    df=pd.read_csv('./DFT/result.csv')
    solv=df[df['idx']==idx]['SolvationEnergy(Hart.)'].values[0]
    return solv

def get_scscore(smiles,tar=['sascore','scscore','spatial']):
    time.sleep(0.05)
    api_url = 'https://askcos.mit.edu/api/molecular-complexity/call-async'
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',
             "Accept": "application/json",
            "Content-Type": "application/json"
            }
    request_data = {
        "smiles": smiles,
        "complexity_metrics": tar
    }
    print(f'SCScore: Processing {smiles}')
    re1 = rq.post(api_url, json=request_data,headers=headers)
    if re1.status_code == 200:
        re2=rq.get(f'https://askcos.mit.edu/api/legacy/celery/task/{re1.json()}')
        if re2.status_code == 200:
            print(f"Post sucess: {re2.status_code}")
            data=json.loads(re2.text)
            state = data["state"]
            result =data["output"]["result"]
            if state == "SUCCESS":
                print(f"Sucess: {re2.status_code}")
                return [float(result[i]) for i in tar]
            else:
                print(f"Failed: {re2.status_code}")
                return [None for i in tar]
        else:
            print(f"Get failed: {re2.status_code}")
            return [None for i in tar]
    else:
        print(f"Post failed: {re1.status_code}")
        return [None for i in tar]


def get_specific_capacity(smiles):
    def ck_Natype(smiles=smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        na_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Na']
        na_idx_map = {idx: i for i, idx in enumerate(na_indices)}

        pattern = Chem.MolFromSmarts("[Na]-O-B-O-[Na]")
        matches = mol.GetSubstructMatches(pattern)
        na_types = [1] * len(na_indices)
        
        if matches:
            na1, _, _, _, na2 = matches[0]
            if na1 in na_idx_map:
                na_types[na_idx_map[na1]] = 0
            if na2 in na_idx_map:
                na_types[na_idx_map[na2]] = 1

        return na_types
    na_types=ck_Natype()
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    mol_weight = Descriptors.MolWt(mol)
    Na_num = sum(1 for i in na_types if i ==1)
    # Na_num = len(na_types)
    specific_capacity=Na_num*96500/(3.6*mol_weight)
    return specific_capacity


def get_battery_freindless(smiles,elemnts):
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    count={}
    rank_list={'N':0, 'Na':0, 'O':0, 'S':0, 'P':0, 'B':1, 'F':0, 'Br':-1, 'H':0, 'C':0, 'Cl':-1}
    for element in  elemnts:
        mass_percent= sum(atom.GetMass() for atom in mol.GetAtoms() if atom.GetSymbol() == element)/sum(atom.GetMass() for atom in mol.GetAtoms())
        count[element]=mass_percent
    rank=sum([rank_list[i]*count[i] for i in elemnts])
    return count,rank

def get_B_wt(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.AddHs(mol)
    mol_weight = Descriptors.MolWt(mol)
    B_num=sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'B')
    return float(100*B_num*10.811/mol_weight)

def get_anode_limit_score(al,min,max):
    if min <= float(al) <= max:
        return 1
    else:
        return 0

#-----------------------------------------------------
# Data analyze
#-----------------------------------------------------

def get_confusion_matrix(dict_hc=None, dict_kmeans=None):
    if dict_hc ==None:
        # dict_hc={'Cluster0': [0, 2, 3, 6, 8, 17, 36, 47, 57, 71, 75, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 334, 337, 340, 342, 368, 388, 391, 404, 502, 547, 548, 555, 558, 559, 560, 564],
        #         'Cluster1': [1, 10, 14, 15, 16, 22, 24, 25, 26, 27, 30, 33, 40, 41, 49, 50, 60, 77, 83, 85, 86, 88, 91, 97, 98, 99, 101, 108, 112, 116, 120, 121, 135, 137, 139, 142, 151, 170, 174, 187, 188, 192, 193, 204, 205, 210, 226, 233, 244, 256, 257, 276, 279, 282, 283, 284, 320, 332, 335, 346, 348, 349, 384, 506, 507, 574, 575, 577, 578, 579, 580, 581],
        #         'Cluster2': [4, 11, 13, 66, 109, 124, 131, 141, 150, 373],
        #         'Cluster3': [5, 115, 123, 126, 222, 252, 253, 263, 264, 277, 278, 287, 292, 294, 301, 308, 309, 311, 315, 318, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
        #         'Cluster4': [7, 19, 37, 48, 52, 72, 76, 78, 143, 169, 218, 232, 255, 286, 304, 305, 306, 314, 347, 356, 364, 367, 370, 389, 390, 392, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 441, 442, 443, 450, 452, 455, 456, 462, 463, 464, 471, 473, 474, 475, 476, 491, 492, 517, 518, 521, 527, 528, 529, 530, 532, 533, 534, 535, 537, 542, 543, 567, 569, 572, 573],
        #         'Cluster5': [9, 56, 59, 106, 114, 119, 125, 128, 165, 185, 200, 202, 214, 239, 249, 268, 275, 280, 281, 295, 296, 321, 323, 324, 326, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557],
        #         'Cluster6': [12, 21, 23, 38, 53, 61, 79, 81, 82, 87, 89, 93, 118, 132, 133, 136, 144, 147, 148, 172, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 242, 266, 270, 271, 272, 273, 289, 297, 303, 310, 316, 325, 344, 352, 405, 444, 445, 446, 447, 448, 453, 454, 457, 458, 459, 460, 495, 497, 504, 508, 509, 510, 511, 522, 545],
        #         'Cluster7': [18, 20, 39, 51, 58, 65, 67, 73, 84, 107, 117, 130, 138, 203, 236, 243, 254, 261, 317, 341, 365, 369, 398, 410, 434, 436, 449, 451, 461, 489, 490, 493, 494, 496, 498, 499, 505, 536, 540, 544, 549, 550, 570, 571, 576],
        #         'Cluster8': [28, 29, 42, 43, 54, 55, 80, 90, 94, 95, 96, 110, 122, 149, 175, 176, 177, 182, 183, 245, 246, 293, 312, 330, 345, 350, 351, 375, 376, 377, 395, 465, 466, 467, 477, 523, 538, 546, 551, 552, 554, 584],
        #         'Cluster9': [31, 32, 34, 35, 44, 45, 46, 62, 63, 100, 102, 103, 104, 105, 111, 113, 134, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 178, 179, 195, 196, 206, 212, 227, 235, 247, 248, 258, 262, 267, 291, 302, 307, 313, 331, 353, 354, 355, 372, 378, 379, 380, 381, 382, 383, 393, 396, 401, 402],
        #         'Cluster10': [64, 68, 69, 70, 74, 127, 129, 140, 164, 166, 167, 186, 199, 208, 209, 219, 220, 221, 237, 240, 241, 259, 319, 333, 336, 338, 339, 357, 358, 363, 403, 406, 407, 409, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 435, 437, 438, 439, 468, 469, 472, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 500, 501, 512, 513, 514, 515, 516, 519, 520, 524, 525, 526, 531, 539, 541, 561, 562, 568],
        #         'Cluster11': [92, 145, 146, 152, 171, 173, 194, 223, 228, 229, 230, 231, 234, 265, 269, 288, 298, 299, 300, 322, 394, 400]
        #         }
        dict_hc={'Cluster0': [64, 68, 69, 70, 74, 127, 129, 140, 164, 166, 167, 186, 199, 208, 209, 219, 220, 221, 237, 240, 241, 259, 319, 333, 336, 338, 339, 357, 358, 363, 403, 406, 407, 409, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 435, 437, 438, 439, 468, 469, 472, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 500, 501, 512, 513, 514, 515, 516, 519, 520, 524, 525, 526, 531, 539, 541, 561, 562, 568],
                'Cluster1': [28, 29, 42, 43, 54, 55, 80, 90, 94, 95, 96, 110, 122, 149, 175, 176, 177, 182, 183, 245, 246, 293, 312, 330, 345, 350, 351, 375, 376, 377, 395, 465, 466, 467, 477, 523, 538, 546, 551, 552, 554, 584],
                'Cluster2': [18, 20, 39, 51, 58, 65, 67, 73, 84, 107, 117, 130, 138, 203, 236, 243, 254, 261, 317, 341, 365, 369, 398, 410, 434, 436, 449, 451, 461, 489, 490, 493, 494, 496, 498, 499, 505, 536, 540, 544, 549, 550, 570, 571, 576],
                'Cluster3': [7, 19, 37, 48, 52, 72, 76, 78, 143, 169, 218, 232, 255, 286, 304, 305, 306, 314, 347, 356, 364, 367, 370, 389, 390, 392, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 441, 442, 443, 450, 452, 455, 456, 462, 463, 464, 471, 473, 474, 475, 476, 491, 492, 517, 518, 521, 527, 528, 529, 530, 532, 533, 534, 535, 537, 542, 543, 567, 569, 572, 573],
                'Cluster4': [1, 10, 14, 15, 16, 22, 24, 25, 26, 27, 30, 33, 40, 41, 49, 50, 60, 77, 83, 85, 86, 88, 91, 97, 98, 99, 101, 108, 112, 116, 120, 121, 135, 137, 139, 142, 151, 170, 174, 187, 188, 192, 193, 204, 205, 210, 226, 233, 244, 256, 257, 276, 279, 282, 283, 284, 320, 332, 335, 346, 348, 349, 384, 506, 507, 574, 575, 577, 578, 579, 580, 581],
                'Cluster5': [5, 115, 123, 126, 222, 252, 253, 263, 264, 277, 278, 287, 292, 294, 301, 308, 309, 311, 315, 318, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
                'Cluster6': [12, 21, 23, 38, 53, 61, 79, 81, 82, 87, 89, 93, 118, 132, 133, 136, 144, 147, 148, 172, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 242, 266, 270, 271, 272, 273, 289, 297, 303, 310, 316, 325, 344, 352, 405, 444, 445, 446, 447, 448, 453, 454, 457, 458, 459, 460, 495, 497, 504, 508, 509, 510, 511, 522, 545],
                'Cluster7': [0, 2, 3, 6, 8, 17, 36, 47, 57, 71, 75, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 334, 337, 340, 342, 368, 388, 391, 404, 502, 547, 548, 555, 558, 559, 560, 564],
                'Cluster8': [9, 56, 59, 106, 114, 119, 125, 128, 165, 185, 200, 202, 214, 239, 249, 268, 275, 280, 281, 295, 296, 321, 323, 324, 326, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557],
                'Cluster9': [31, 32, 34, 35, 44, 45, 46, 62, 63, 100, 102, 103, 104, 105, 111, 113, 134, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 178, 179, 195, 196, 206, 212, 227, 235, 247, 248, 258, 262, 267, 291, 302, 307, 313, 331, 353, 354, 355, 372, 378, 379, 380, 381, 382, 383, 393, 396, 401, 402],
                'Cluster10': [92, 145, 146, 152, 171, 173, 194, 223, 228, 229, 230, 231, 234, 265, 269, 288, 298, 299, 300, 322, 394, 400],
                'Cluster11': [4, 11, 13, 66, 109, 124, 131, 141, 150, 373]
                }
        # dict_hc = dict(sorted(dict_hc.items()))
    if dict_kmeans ==None:
        dict_kmeans={'Cluster0': [37, 64, 127, 129, 164, 166, 167, 169, 199, 208, 209, 219, 220, 221, 240, 241, 287, 315, 325, 333, 336, 356, 357, 358, 363, 367, 403, 406, 409, 411, 415, 416, 417, 418, 420, 421, 423, 425, 429, 430, 431, 438, 441, 442, 443, 468, 469, 472, 474, 475, 478, 479, 486, 488, 500, 504, 512, 513, 514, 515, 516, 520, 521, 524, 526, 529, 531, 534, 539, 561, 562, 567],
                    'Cluster1': [18, 19, 20, 28, 29, 39, 42, 43, 51, 54, 55, 84, 90, 94, 95, 96, 110, 117, 119, 122, 134, 138, 149, 182, 183, 203, 243, 245, 282, 293, 330, 345, 347, 350, 351, 369, 375, 376, 377, 395, 401, 449, 451, 461, 465, 466, 467, 496, 498, 499, 536, 537, 538, 544, 546, 549, 550, 551, 552, 554, 576, 584],
                    'Cluster2': [67, 68, 74, 106, 130, 140, 186, 236, 237, 254, 259, 261, 310, 319, 338, 339, 365, 398, 407, 410, 412, 413, 414, 434, 435, 436, 437, 439, 480, 481, 482, 483, 484, 485, 487, 489, 490, 492, 493, 494, 501, 505, 519, 525, 540, 541, 545, 568],
                    'Cluster3': [7, 48, 52, 58, 65, 72, 73, 76, 107, 218, 255, 286, 305, 314, 317, 324, 341, 364, 408, 419, 422, 424, 426, 427, 428, 432, 433, 440, 455, 471, 473, 476, 491, 517, 518, 527, 528, 530, 532, 533, 535, 542, 543, 569, 570, 571, 572, 573],
                    'Cluster4': [1, 16, 22, 24, 25, 26, 27, 30, 31, 33, 41, 60, 62, 86, 88, 91, 97, 98, 99, 101, 108, 112, 120, 121, 135, 142, 150, 151, 174, 187, 188, 192, 193, 204, 205, 206, 210, 226, 233, 244, 256, 257, 276, 279, 283, 284, 335, 346, 378, 384, 574, 578, 579, 580, 581],
                    'Cluster5': [5, 115, 123, 126, 214, 222, 223, 249, 252, 253, 263, 264, 265, 268, 269, 275, 277, 278, 292, 294, 295, 296, 301, 308, 309, 311, 318, 326, 327, 328, 329, 343, 359, 362, 371, 374, 385, 563, 565, 566, 582, 583],
                    'Cluster6': [12, 21, 23, 38, 53, 61, 78, 79, 80, 81, 82, 87, 89, 93, 118, 132, 133, 136, 143, 144, 147, 148, 172, 175, 176, 177, 180, 181, 189, 190, 191, 211, 213, 216, 224, 225, 232, 242, 246, 266, 270, 271, 272, 273, 289, 297, 303, 304, 306, 312, 316, 344, 352, 370, 389, 390, 392, 393, 405, 444, 445, 446, 447, 448, 450, 452, 453, 454, 456, 457, 458, 459, 460, 462, 463, 464, 477, 495, 497, 508, 509, 510, 511, 522, 523],
                    'Cluster7': [0, 2, 3, 6, 8, 10, 14, 17, 36, 40, 47, 49, 57, 69, 70, 71, 75, 77, 116, 168, 184, 197, 198, 201, 207, 215, 217, 238, 250, 251, 260, 274, 285, 290, 320, 334, 337, 340, 342, 348, 388, 404, 502, 506, 507, 547, 548, 555, 558, 559, 560, 564],
                    'Cluster8': [9, 15, 50, 56, 59, 83, 85, 114, 125, 128, 137, 139, 165, 170, 185, 200, 202, 239, 280, 281, 321, 323, 332, 349, 360, 361, 366, 386, 387, 397, 399, 470, 503, 553, 556, 557, 575, 577],
                    'Cluster9': [32, 34, 35, 44, 45, 46, 63, 100, 102, 104, 105, 111, 113, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 178, 179, 195, 196, 212, 227, 247, 248, 258, 262, 267, 291, 307, 313, 331, 353, 354, 355, 372, 379, 380, 381, 382, 383, 396, 402],
                    'Cluster10': [92, 103, 145, 146, 152, 158, 171, 173, 194, 228, 229, 230, 231, 234, 235, 288, 298, 299, 300, 302, 322, 394, 400],
                    'Cluster11': [4, 11, 13, 66, 109, 124, 131, 141, 368, 373, 391]
                }
        
    
    groups1 = list(dict_hc.keys())
    groups2 = list(dict_kmeans.keys())
    matrix = np.zeros((len(groups1), len(groups2)), dtype=int)
    # Calculate Rand index
    for i, g1 in enumerate(groups1):
        for j, g2 in enumerate(groups2):
            intersection = set(dict_hc[g1]) & set(dict_kmeans[g2])
            matrix[i][j] = len(intersection)
    df = pd.DataFrame(matrix, index=groups1, columns=groups2)

    total_samples = np.sum(matrix)
    a = 0
    for value in matrix.flatten():
        if value > 1:
            a += (value * (value - 1)) // 2 
    total_pairs = (total_samples * (total_samples - 1)) // 2
    b = total_pairs - np.sum([(np.sum(row) * (np.sum(row) - 1)) // 2 for row in matrix]) \
        - np.sum([(np.sum(col) * (np.sum(col) - 1)) // 2 for col in matrix.T]) + a
    rand_index = (a + b) / total_pairs
    print(f'Rand index: {rand_index}')
    return df

def get_sctter_info_for_origin(df,save):
    grouped = df.groupby('Cluster')
    new_dfs = []
    for group_name, group_data in grouped:
        group_data = group_data.drop(columns=['Cluster'])
        group_data.columns = [f"{group_name}_{col}" for col in group_data.columns]
        new_dfs.append(group_data.reset_index(drop=True))
    result = pd.concat(new_dfs, axis=1)
    result.to_csv(save,index=False)
    # print(result)
    return result   

def get_rank_data(df,input_path):
    df['battery_elemetal_friendness']=df['canonicalsmiles'].apply(lambda x:get_battery_freindless(x,elemnts=get_all_elements(df=pd.read_csv('./data/data.csv')))[1])
    df['capacity(mAh/g)']=df['canonicalsmiles'].apply(lambda x :get_specific_capacity(x))
    df=df.reset_index(drop=True)
    record=[]
    for smiles in df['canonicalsmiles'].to_list():
        data=get_scscore(smiles,tar=['scscore','sascore','spatial'])
        scs,sas,spacail=data[0],data[1],data[2]
        record.append({'scscore':scs,'sascore':sas,'spacial_score':spacail})
        # print(record)
    
    df2=pd.DataFrame(record)
    df2=df2.reset_index(drop=True)
    df3=pd.concat([df,df2],ignore_index=True)
    return df3

def rank(df):
    # df=df[df['scscore']<]
    w=0.2
    cols=['battery_elemetal_friendness','sascore','scscore','capacity(mAh/g)','spacial_score']
    df[cols] = df[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # df['total_score']=-(df['sascore']+df['scscore']/2)+df['capacity(mAh/g)']+df['B_wt(%)']
    # df['sascore']=df['sascore'].apply(lambda x: discretize(x,0.1))
    # df['scscore']=df['scscore'].apply(lambda x: discretize(x,1))
    df['total_score']=-(df['scscore']*df['spacial_score'])*(w*df['capacity(mAh/g)']+(1-w)*df['battery_elemetal_friendness'])
    df=df.sort_values(by='total_score',ascending=False)
    return df


def kde_sampling_order(df, bandwidth=0.1):
    np.random.seed(42)
    from sklearn.neighbors import KernelDensity
    from sklearn.metrics import pairwise_distances_argmin_min
    df_array = df.values
    n_samples = len(df)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(df_array)

    sampled = kde.sample(n_samples)

    hit_set = set()
    hit_order = []

    for s in sampled:
        idx, _ = pairwise_distances_argmin_min([s], df_array)
        index = idx[0]
        if index not in hit_set:
            hit_set.add(index)
            hit_order.append(index)
        if len(hit_order) == n_samples:
            break
    return hit_order

def find_product_solvenergy(product):
    # Hartree
    if False:
        d= {'FC(F)(F)SSC(F)(F)F':9999,
            'CCCSSCCC':9999,
            'CCOCCCCOCC':9999,
            'COOC':9999,
            'CCCCCSSCCCCC':9999, 
            'CCCOCOOCOCCC':9999, 
            'CCCCCCCCCC':9999, 
            'COCOOCOC':9999, 
            'COCCCCCCOC':9999, 
            'CCOCOOCOCC':9999, 
            'CC':9999, 
            'FC(F)(F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)C(F)(F)F':9999, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':9999, 
            'CCOCCOCC':9999, 
            'CCCCOOCCCC':9999, 
            'O=NC(F)(F)OC(F)(F)F':9999, 
            'CCSSCC':9999, 
            'COCSSCOC':9999, 
            'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F':9999, 
            'O=[N+]([O-])C(F)(F)F':9999, 
            'CCCOCSSCOCCC':9999, 
            'CCCCCCCC':9999, 
            'CCCCCOOCCCCC':9999, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':9999, 
            'CCCC':9999, 
            'CCCCCC':9999, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F':9999, 
            'COCCOC':9999, 
            'COCCOOCCOC':9999, 
            'CCCCSSCCCC':9999, 
            'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F':9999, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':9999, 
            'CCCOCCOCCC':9999, 
            'O=NC(F)(F)F':9999, 
            'COCCCSSCCCOC':9999, 
            'CCOCCOOCCOCC':9999, 
            'CCOCSSCOCC':9999, 
            'COCCCCOC':9999}
        df=pd.read_csv('./DFT/result.csv')
        index_dict=joblib.load('./DFT/product_index.pkl')
        df['smi'] = df['idx'].apply(lambda idx: index_dict.get(idx, None))
        for k,v in d.items():
            if k !=None:
                solv=df[df['smi']==k]['SolvationEnergy(Hart.)'].values[0]
                d[k]=solv
    # print(d)
    if True:
        d={'FC(F)(F)SSC(F)(F)F': 0.0051292999999077,
            'CCCSSCCC': -0.0024166999999124,
            'CCOCCCCOCC': -0.0059598999999934,
            'COOC': -0.0041785999999888,
            'CCCCCSSCCCCC': -0.0025198999999247,
            'CCCOCOOCOCCC': -0.0106616000001622, 
            'CCCCCCCCCC': 0.0023680999999555, 
            'COCOOCOC': -0.0105417999999986, 
            'COCCCCCCOC': -0.006707399999982, 
            'CCOCOOCOCC': -0.0093408000001318, 
            'CC': 0.0022181000000074, 
            'FC(F)(F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)C(F)(F)F': 0.0139096999992034, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F': 0.013786699999855, 
            'CCOCCOCC': -0.0054144000000633, 
            'CCCCOOCCCC': -0.0045862000000056, 
            'O=NC(F)(F)OC(F)(F)F': 0.0024647000000186, 
            'CCSSCC': -0.0027323000000478, 
            'COCSSCOC': -0.0064932000000226, 
            'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F': 0.0111092000001917, 
            'O=[N+]([O-])C(F)(F)F': 0.0030142999999043, 
            'CCCOCSSCOCCC': -0.006530899999916,
            'CCCCCCCC': 0.0024710999999797,
            'CCCCCOOCCCCC': -0.004780299999993, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F': 0.0112294999998994, 
            'CCCC': 0.0019637000000329, 
            'CCCCCC': 0.0021026000000006, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F': 0.009126599999945, 
            'COCCOC': -0.0066190000000574, 
            'COCCOOCCOC': -0.0118097999999235, 
            'CCCCSSCCCC': -0.0022926000001461, 
            'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F': 0.0136694000002535, 
            'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F': 0.0167071000005307, 
            'CCCOCCOCCC': -0.0059125000000221, 
            'O=NC(F)(F)F': 0.0001050000000759, 
            'COCCCSSCCCOC': -0.0099835000000894, 
            'CCOCCOOCCOCC': -0.0142865999999912, 
            'CCOCSSCOCC': -0.0075817000001734, 
            'COCCCCOC': -0.006437700000049}

    return d[product]


def find_product_mp(product):
    # ℃, Data from Reaxy
    d= {'FC(F)(F)SSC(F)(F)F':125,
        'CCCSSCCC':8,
        'CCOCCCCOCC':-165.5,
        'COOC':-105,
        'CCCCCSSCCCCC':8, 
        'CCCOCOOCOCCC':9999, 
        'CCCCCCCCCC':-29.67, 
        'COCOOCOC':9999, 
        'COCCCCCCOC':9999, 
        'CCOCOOCOCC':9999, 
        'CC':-182.78, 
        'FC(F)(F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)C(F)(F)F':9999, 
        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':-25, 
        'CCOCCOCC':-74, 
        'CCCCOOCCCC':9999, 
        'O=NC(F)(F)OC(F)(F)F':9999, 
        'CCSSCC':-120, 
        'COCSSCOC':9999, 
        'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F':9999, 
        'O=[N+]([O-])C(F)(F)F':-30, # Boiling point, its m.p. not find, but must < -30 
        'CCCOCSSCOCCC':9999, 
        'CCCCCCCC':-56.81, 
        'CCCCCOOCCCCC':9999, 
        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':-70, 
        'CCCC':-138.34, 
        'CCCCCC':-95.6, 
        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F':-128.19, 
        'COCCOC':-58, 
        'COCCOOCCOC':9999, 
        'CCCCSSCCCC':-138, 
        'FC(F)(F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(F)(F)F':9999, 
        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F':32, 
        'CCCOCCOCCC':9999, 
        'O=NC(F)(F)F':-196, 
        'COCCCSSCCCOC':9999, 
        'CCOCCOOCCOCC':9999, 
        'CCOCSSCOCC':9999, 
        'COCCCCOC':9999}
    return d[product]
    # pass


def find_anode_limt(smiles):
    df=pd.read_csv('./DFT/voltage/result_voltage.csv')
    try:
        voltage=df[df['smiles']==smiles]['Voltage'].values[0]
    except:
        voltage=9999
    return voltage

if __name__ == '__main__':
    # move_volt_to_data()
    
    pass
    # print(kde_sampling_order(pd.read_csv('./outputs/features.csv')))
    # def get_name(name):
    #     match=re.match(r'^mol(\d+)_',name)
    #     if match:
    #         mol=match.group(1)
    #         return int(mol)
    # print(get_name('mol0_'))

import pandas as pd
class Pareto:
    def __init__(self, df,opt_target:dict):
        self.target_cols=[k for k,v in opt_target.items()]
        self.opt_direction=[v for k,v in opt_target.items()]

        def change_derection(values:list,direction:list):
            return [values[i] if direction[i] == 'min' else -values[i] for i in range(len(values))]
        population = []
        info_cols=[i for i in df.columns if i not in self.target_cols]
        for idx, row in df.iterrows():
            target_values = change_derection([row[col] for col in self.target_cols],self.opt_direction)
            info=[row[i] for i in info_cols]
            # old_idx=row[info_cols[0]]
            # smiles=row[info_cols[1]]
            # cluster=row[info_cols[2]]
            # population.append((idx, target_values,[old_idx,smiles,cluster]))
            population.append((idx, target_values,info))

        self.population = population
        # self.target_cols=target_cols
        self.info_cols=info_cols

    
    def dominate(self, a, b):
        # Defluat opt direction > min()
        a=a[1] # a=(idx,[target1,target2,target3],info_list)
        b=b[1] # b=(idx,[target1,target2,target3],info_list)
        compare= [None]*len(a)
        for i in range(len(a)):  
            if b[i] < a[i]:
                compare[i]='b' # better
            elif b[i] > a[i]:
                compare[i]='w' # worse
            else:
                compare[i]='=' # equal
        if 'b' in compare and 'w' not in compare:
            return True
        else:
            return False

    def pareto_front(self):

        pareto_front = []
        for i in range(len(self.population)):
            is_dominated = False
            for j in range(len(self.population)):
                if i != j and self.dominate(self.population[i], self.population[j]):  # 检查 population[j] 是否支配 population[i]
                    is_dominated = True #如果支配了，[i]必定不在前沿，故break
                    break
            if not is_dominated:
                pareto_front.append(self.population[i])
        df = self.population_to_df(population=pareto_front)
        return df

    def population_to_df(self,population):
        def return_derection(values:list,direction:list):
            return [values[i] if direction[i] == 'min' else -values[i] for i in range(len(values))]
        records = []
        for idx, target_list, info in population:
            row={}
            row.update({k: v for k, v in zip(self.target_cols, return_derection(target_list,direction=self.opt_direction))})
            for i in range(len(info)):
                row[self.info_cols[i]] = info[i]
            # row[self.info_cols[0]] = info[0]
            # row[self.info_cols[1]] = info[1]
            # row[self.info_cols[2]] = info[2]
            records.append(row)
        return pd.DataFrame(records)

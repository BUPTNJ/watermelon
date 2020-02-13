import pandas as pd
from scipy.stats import norm
def make_bayes(filename,discrete_attr_list,continuous_attr_list,label_attr):
    data = pd.read_csv(filename)
    prob={}
    vtype={}
    continuous_dict={}
    for discrete_attr in discrete_attr_list:
        vtype[discrete_attr]=data[discrete_attr].value_counts().shape[0]
    label_groups=data.groupby(label_attr)
    for label_group in label_groups:
        for continuous_attr in continuous_attr_list:
            continuous_dict[(label_group[0],continuous_attr)]=(label_group[1][continuous_attr].mean(),label_group[1][continuous_attr].std())
        prob[label_group[0]]=(label_group[1].shape[0]+1)/(data.shape[0]+len(label_groups))
        for discrete_attr in discrete_attr_list:
            total_count=label_group[1].shape[0]
            disc_groups=label_group[1].groupby([discrete_attr])
            for disc_group in disc_groups:
                prob[(label_group[0],disc_group[0])]=((disc_group[1][discrete_attr].count()+1)/(total_count+vtype[discrete_attr]))
    return prob,continuous_dict
def predict(prob,continuous_dict,dict_attr_dict,contin_attr_dict):
    result={'是':prob['是'],'否':prob['否']}
    for attr in dict_attr_dict.keys():
        result['是']*=prob[('是',dict_attr_dict[attr])]
        result['否']*=prob[('否',dict_attr_dict[attr])]
    for attr in contin_attr_dict.keys():
        result['是']*=norm.pdf(contin_attr_dict[attr], continuous_dict[('是',attr)][0], continuous_dict[('是',attr)][1])
        result['否']*=norm.pdf(contin_attr_dict[attr], continuous_dict[('否',attr)][0], continuous_dict[('否',attr)][1])
    return result
prob,continuous_dict=make_bayes('data.csv',['色泽','根蒂','敲声','纹理','脐部','触感'],['密度','含糖率'],'好瓜')
result=predict(prob,continuous_dict,{'色泽':'青绿','根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑'},{'密度':0.697,'含糖率':0.460})
print(result)
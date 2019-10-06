import pandas as pd
import numpy as np
from numpy import log2 as log
eps = np.finfo(float).eps

dataset={'Outlook':['play','play','stay','play','stay','play','stay','play','play','play','stay','stay','play'],
         'Temperature':['weak','strong','weak','weak','strong','weak','strong','weak','weak','strong','weak','strong','weak'],
         'Humidity':['hot','hot','hot','mid','cold','cold','cold','mid','cold','mid','mid','mid','hot'],
         'Wind':['sunny','sunny','rain','overcast','rain','overcast','rain','sunny','sunny','overcast','sunny','rain','overcast'],
         'Play_Golf':['high','high','high','high','normal','normal','normal','normal','normal','normal','high','high','normal']}

df = pd.DataFrame(dataset,columns=['Outlook','Temperature','Humidity','Wind','Play_Golf'])


def buildTree(df, tree=None):
    Class = df.keys()[-1] 
    node = find_winner(df)
    attValue = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValue:
        subtable = df[df[node] == value].reset_index(drop=True)
        clValue, counts = np.unique(subtable['Outlook'], return_counts=True)
        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Calling the function recursively
    return tree

def find_winner(df):
    Entropy_att = [] 
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]

def find_entropy(df):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy

def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]  # To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    variables = df[
        attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)

tree =buildTree(df)
import pprint
pprint.pprint(tree)

import csv
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import ipywidgets as widgets
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
df = pd.read_csv("clean.csv",header=None,low_memory=False)
df.columns=['CardNo','ProductId','Description', 'Quantity']
df=df.drop(df.index[0])
global k,metric
k=6
metric='cosine'
print df.head()

basket = (df.groupby(['CardNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('CardNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    print '{0} most similar users for User {1}:\n'.format(k,user_id)
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;

        else:
            print '{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i])
            
    return similarities,indices

uid=12


def predict_userbased(user_id, col, ratings, metric = metric, k=k):
    prediction=[]
    
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_id-1,:].mean() #to adjust for zero based indexing
    
    for c in range(0,len(col)):
        sum_wt = np.sum(similarities)-1
        product=1
        wtd_sum = 0 
        pred=0
        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i]+1 == user_id:
                continue;
            else: 
                ratings_diff = ratings.iloc[indices.flatten()[i],c]-np.mean(ratings.iloc[indices.flatten()[i],:])
                product = ratings_diff * (similarities[i])
                wtd_sum = wtd_sum + product
    
        pred = float(round(mean_rating + (wtd_sum/sum_wt)))
        prediction.append(pred)
    #print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)
    #print prediction
    return prediction


p=predict_userbased(uid,basket_sets.columns,basket_sets);

pro=list(basket_sets.iloc[uid-1])

names=[basket_sets.columns[index] for index,v in enumerate(pro) if v==1.0]
print "Items consumed by user {0}\n".format(uid)
print names


ind=[index for index, v in enumerate(p) if v == 1.0]
print "Items recommended to user {0}\n".format(uid)
for k in ind:
    print basket_sets.columns[k]



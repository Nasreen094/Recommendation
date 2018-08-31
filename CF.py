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
k=4
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

#USER BASED
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

similarities,indices = findksimilarusers(1,basket_sets, metric='cosine')


def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_id-1,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_id-1]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)

    return prediction

def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]    
    ratings=ratings.T
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[item_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    print '{0} most similar items for item {1}:\n'.format(k,item_id)
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue;

        else:
            print '{0}: Item {1} :, with similarity of {2}'.format(i,indices.flatten()[i]+1, similarities.flatten()[i])


    return similarities,indices

def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue;
        else:
            product = ratings.iloc[user_id-1,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product                              
    prediction = int(round(wtd_sum/sum_wt))
    print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)      

    return prediction

def recommendItem(user_id, item_id, ratings):
    
    if user_id<1 or user_id>70 or type(user_id) is not int:
        print 'Userid does not exist. '
    else: 
        
        ids = {0:'User-based CF (cosine)',1:'User-based CF (correlation)',2:'Item-based CF (cosine)'
              }
        val=int(raw_input('select'+str(ids)))
           
        if (val==0):
                    metric = 'cosine'
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
                    
        elif (val==1):                       
                    metric = 'correlation'               
                    prediction = predict_userbased(user_id, item_id, ratings, metric)
        elif (val==2):
                    prediction = predict_itembased(user_id, item_id, ratings)
        else:
                    print "INVALID ENTRY"
        
    
        if ratings.iloc[user_id-1][item_id-1] == 1: 
                    print 'Item already rated'
        else:
            if prediction==1:
                print '\nItem recommended'
            else:
                print 'Item not recommended'

uid=int(raw_input("Enter user id(1-70)")) 

iid=int(raw_input("Enter item id"))       
print recommendItem(uid,iid,basket_sets)




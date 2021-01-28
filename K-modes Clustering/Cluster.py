#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import random


# In[12]:


#global variables used after
modeframe=pd.DataFrame()
#already 1 for sample
iteration=1
answerframe=pd.DataFrame()


# In[13]:


# data=pd.read_csv("agaricus-lepiota.data" , header=None)


# In[14]:


# data=pd.DataFrame(data.values, columns = ['label','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','surface-above-ring','surface-below-ring','color-above-ring','color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])


# In[15]:


def clean_data(data):
#     print(data["stalk-root"].unique())

    #veil type can be removed as it plays no part
    #creating a new data frame which doesnt have '?'
    dataNew=data[data['stalk-root']!='?']
    #you can comment out the below line to see the correlation I found
#     print(dataNew.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson')['stalk-root'])
    # From this I figured that habitat effects the stalk-root the most.

    # I plotted to find out that which information I can fill with habitat
    plt.scatter(data['stalk-root'], data['habitat'])
    plt.xlabel("stalk-root")
    plt.ylabel("habitat")
    plt.show()

    #Except w , each has a relation with stalk-root, Now I have ratio of each stalk root for certain habitats
    dataRatio=dataNew.groupby("habitat")["stalk-root"].value_counts(normalize=True).mul(100)
#     print(dataRatio)

    #These loops will fill data according to dataRatio of stalk roots present for a certain habitat.
    for i in range(len(data)):
        if data['stalk-root'][i]=='?':
            if data['habitat'][i]=='l':
                number=random.randint(0, 100)
                if number >= dataRatio[6]:
                    data['stalk-root'][i]='c'
                else:
                    data['stalk-root'][i]='b'

    for i in range(len(data)):
        if data['stalk-root'][i]=='?':
            if data['habitat'][i]=='p':
                number=random.randint(0, 100)
                if number >= dataRatio[10]:
                    data['stalk-root'][i]='r'
                else:
                    data['stalk-root'][i]='b'

    for i in range(len(data)):
        if data['stalk-root'][i]=='?':
            if data['habitat'][i]=='d':
                number=random.randint(0, 100)
                if number >= dataRatio[0]:
                    data['stalk-root'][i]='c'
                else:
                    data['stalk-root'][i]='b'

    for i in range(len(data)):
        if data['stalk-root'][i]=='?':
            if data['habitat'][i]=='g':
                number=random.randint(0, 100)
                if number <= dataRatio[2]:
                    data['stalk-root'][i]='e'
                elif number <= dataRatio[2]+dataRatio[3]:
                    data['stalk-root'][i]='b'
                elif number <= dataRatio[2]+dataRatio[3]+dataRatio[4]:
                    data['stalk-root'][i]='c'
                else:
                    data['stalk-root'][i]='r'

    #The next in line was population in correlation
    plt.scatter(data['stalk-root'], data['population'])
    plt.xlabel("stalk-root")
    plt.ylabel("population")
    plt.show()

    #After we have filled the data with the help of 'habitat', we use 'population' for this purpose
    dataRatio=dataNew.groupby("population")["stalk-root"].value_counts(normalize=True).mul(100)
#     print(dataRatio)

    for i in range(len(data)):
        if data['stalk-root'][i]=='?':
            if data['population'][i]=='c':
                number=random.randint(0, 100)
                if number >= dataRatio[1]:
                    data['stalk-root'][i]='b'
                else:
                    data['stalk-root'][i]='c'

    return data


# In[16]:


def initial_centroid(data, no_of_clusters):
    rows=len(data)
    columns=len(data.columns)
    dataArray = data.to_numpy()
    #for initial centroids I take random rows from database
    sampleArray= data.sample(n=no_of_clusters)
    sampleArray=sampleArray.to_numpy()
    distances= pd.DataFrame()
    #I form a data frame of distances for each centroid which I got from sample
    for i in range(no_of_clusters):
        distances['distance'+str(i+1)]= get_distance(dataArray,sampleArray[i])

    distances= distances.to_numpy()
    #Now I take minimum of each row, which gives me the cluster which has the minimum distance to each row, I am saving those indexes in a new array
    clusters=[0]*rows
    for i in range(rows):
        clusters[i]=np.argmin(distances[i])

    #Now I form a new list of dataframes, where each element of this list is a dataframe of a single cluster
    arrayDataFrame =[pd.DataFrame(columns = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])]*no_of_clusters
    for i in range(len(data)):
        arrayDataFrame[clusters[i]]=arrayDataFrame[clusters[i]].append(data.iloc[i])

    #You can access each of them by arrayDataFrame[index]
    return arrayDataFrame




# In[17]:


def get_distance(data,centroid):
    rows=len(data)
    columns=len(data[0])
    distances=[0]*rows
#     print(len(data[0]))

    for i in range(rows):
        distance=0
        for j in range(columns):
            if(centroid[j]!=data[i][j]):
                distance=distance+1
        distances[i]=distance
    return distances


# In[18]:


# modeframe=pd.DataFrame()
# iteration=1
# answerframe=pd.DataFrame()
def K_centroids(data, oldframe, no_of_clusters):
    rows=len(data)
    columns=len(data.columns)
    dataArray = data.to_numpy()
    global modeframe
    global iteration
    global answerframe
    iteration = iteration+1
    distances= pd.DataFrame()

    #Now I use mode as my centroids to find distances from each centroid
    for i in range(no_of_clusters):
        modeArray= oldframe[i].mode().dropna()
        modeArray=modeArray.to_numpy()
        distances['distance'+str(i+1)]= get_distance(dataArray,modeArray.all(0))

    distances= distances.to_numpy()
#     print(distances)
    #Finding minimum distance and saving the index for cluster again.
    clusters=[0]*rows
    for i in range(rows):
        clusters[i]=np.argmin(distances[i])
    #print(clusters)
    newframe =[pd.DataFrame(columns = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])]*no_of_clusters
#     print(len(data))

    #Forming new dataframes again from the new centroids
    for i in range(len(data)):
        newframe[clusters[i]]=newframe[clusters[i]].append(data.iloc[i])
#     print(newframe)
    for i in range(no_of_clusters):
        newframe[i]=newframe[i].reset_index(drop=True)

    #I will use mode_combination here to find modes of our new clusters and for a dataframe consisting of these centroids.
    mode_combination = pd.DataFrame()
    for i in range(no_of_clusters):
        mode_combination=mode_combination.append(newframe[i].mode().dropna())
#     print(Modecombination)
    #If the new centroids are equal to our old centroids, we print the number of iterations and form our final cluster
    if(modeframe.equals(mode_combination)):
        print("total iterations",iteration)
        answerframe=newframe
        return newframe
    #Otherwise we save our mode dataframe to compare it in the next case and do recursion to proceed
    else :
        modeframe = mode_combination
        K_centroids(data, newframe,no_of_clusters)





# In[19]:


#simly add the cluster attribute and append dataframes vertically to form the final answer.
def create_csv(no_of_clusters):
    global answerframe
    tempframe=pd.DataFrame()
    for i in range(no_of_clusters):
        answerframe[i]['Cluster'] = i+1
        tempframe=tempframe.append(answerframe[i],ignore_index=True)
    tempframe.to_csv("clusters.csv")



# In[20]:


def main():
#     file = sys.argv[1]
    #The program takes about 3-4 minutes usually, may take a little more in worse cases.
    data=pd.read_csv('agaricus-lepiota.data', header=None)
    data=pd.DataFrame(data.values, columns = ['label','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])
    #cleaning the data
    data= clean_data(data)
    del data['label']
    clusters = 5
    #Initialization of centroids and assigning observations to a cluster.
    frame=initial_centroid(data,clusters)
    #Computing new centroids, assigning observations to new clusters and terminating if old mode is equal to new mode.
    K_centroids(data,frame,clusters)
    #creating a csv file for the final clusters and adding an attribute which represents the cluster number.
    create_csv(clusters)
if __name__ == '__main__':
    main()


# In[ ]:

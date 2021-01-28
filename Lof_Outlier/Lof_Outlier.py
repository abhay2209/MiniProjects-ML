#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


data=pd.read_csv("outliers-3.csv")


# In[3]:


def Eucledian_distance(value1,value2):
    return math.sqrt((value1[0]-value2[0])**2 + (value1[1]-value2[1])**2)



# In[4]:


def all_distance(data):
    distance = np.zeros((len(data), len(data)))
    #First I find all distances from each point, distance[0] represents all distances form first point
    for i in range(len(data)):
        for j in range(len(data)):
                distance[i][j]=Eucledian_distance(data.iloc[i],data.iloc[j])
    return distance


# In[5]:


def sort_distances(distance, data, k):
    sortedDistance = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            sortedDistance[i][j]=distance[i][j]
        #Now we sort all distances so it is easier to find kth distance
        sortedDistance[i].sort()

    return sortedDistance


# In[6]:


def point_finder(sortedDistance, distance, k_hyperparameter):
    findIndex = np.zeros((len(data), k_hyperparameter+1))
    for i in range(len(sortedDistance)):
        for j in range((k_hyperparameter+1)):
            for k in range(len(distance)):
                if sortedDistance[i][j]==distance[i][k]:
                    #Now I find indexes of those k distances so I know which points we are talking about
                    findIndex[i][j]=k
    return findIndex


# In[7]:


def reachabilityDistance(distance, sorted_distance, points, k):
    reachability_distance=np.zeros((len(sorted_distance),len(sorted_distance)))
    # Now I find the reachability distance.
    for i in range(len(distance)):
        for j in range(len(distance)):
                reachability_distance[i][j]=max(distance[i][j],sorted_distance[j][k])
    return reachability_distance


# In[8]:


def Nk_finder(sorted_distance, k):
    #Value of Nk will be atleast k
    Nk = np.full((len(sorted_distance)), k)
    for i in range(len(sorted_distance)):
        value=False
        l=k
        while value==False:
#             print("works")
            #it will be k + points that are on the boundary, hence this loops checks that
            if sorted_distance[i][l]==sorted_distance[i][l+1]:
                Nk[i]=Nk[i]+1
                l=l+1
            else:
                value=True
    return Nk


# In[9]:


def averageReachability(reachability_distance,sorted_distance, points, k, Nk):
    average_reachability=np.zeros(len(sorted_distance))
    for i in range(len(sorted_distance)):
        for j in range(1,k+1):
                average_reachability[i]=average_reachability[i]+reachability_distance[i][int(points[i][j])]
    #             print(reachability_distance[i][int(points[i][j])],j)
        #Now i find the average of the reachability distance.
        #Here I use k instead of Nk as discussed in email
        #My Eucledian distance is in 8 decmilas hence, in almost all cases k=Nk and hence giving the same answer.
        average_reachability[i]=Nk[i]/average_reachability[i]
    return average_reachability


# In[10]:


def lofRatio(average_reachability, points, k, Nk):
    lof_ratio = np.zeros((len(average_reachability)))
    for i in range(len(average_reachability)):
        for j in range(1,k+1):
            lof_ratio[i]=lof_ratio[i]+average_reachability[int(points[i][j])]
    #         print(int(points[i][j]))
        # Now i find the lof ratios
        lof_ratio[i]=lof_ratio[i]/average_reachability[i]
        lof_ratio[i]=lof_ratio[i]/Nk[i]
    return lof_ratio


# In[11]:


def main():
    #You just have to run the code. To try different k,change value below
    k=5
    distance = all_distance(data)
    sorted_distance=sort_distances(distance, data, k)
    points = point_finder(sorted_distance, distance, k)

    Nk=Nk_finder(sorted_distance, k)

    reachability_distance=reachabilityDistance(distance, sorted_distance, points, k)
    average_reachability=averageReachability(reachability_distance,sorted_distance, points, k, Nk)
    lof_ratio=lofRatio(average_reachability, points, k, Nk)


    sum_of_rows = lof_ratio.sum()
    normalized_array = np.zeros((len(lof_ratio)))
    for i in range(len(lof_ratio)):
        normalized_array[i] = lof_ratio[i] / sum_of_rows
#     print(normalized_array)
#     plt.plot(normalized_array)

    data['LOF_ratio']=lof_ratio
    outliers=data[data['LOF_ratio']>1.815]
    cluster=data[data['LOF_ratio']<=1.815]
    plt.plot(outliers['X1'],outliers['X2'],'r.')
    plt.plot(cluster['X1'],cluster['X2'],'b.',)

if __name__ == '__main__':
    main()


# In[ ]:

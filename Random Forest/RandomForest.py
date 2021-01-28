#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import math
from random import *


# In[20]:


def entropyFind(dataLabel):
    #this finds the unique kind of attribute and then the total kinds of it
    attributeKind, count = np.unique(dataLabel, return_counts = True)
    #taking the sum
    totalCount =np.sum(count)
    #finding the entropy
    return np.sum([(-count[i]/totalCount)*np.log2(count[i]/totalCount) for i in range(len(attributeKind))])


# In[21]:


def informationGain(dataset, feature, label ):
   #First we calculate the entropy from data set
    entropyValue = entropyFind(dataset[label])
    values, count= np.unique(dataset[feature],return_counts=True)

    # then we calculate the weighted entropy
    entropyWeighted = np.sum([(count[i]/np.sum(count))*entropyFind(dataset.where(dataset[feature] == values[i]).dropna()[label]) for i in range(len(values))])
    return entropyValue - entropyWeighted



# In[22]:


def createDecisionTree(dataset, df, features, label, parent):
    uniqueData =np.unique(dataset[label])
    dataParent = np.unique(df[label],return_counts=True)
    #print(dataParent[1])
    #we need more than 1 unique to start our split
    if(len(uniqueData))<=1:
        return uniqueData[0]
    #dataset should have information
    elif len(dataset)==0:
        return uniqueData[np.argmax(dataParent[1])]
    #it should consist of atleast 2 features or say attributes
    elif len(features)==0:
        return parent

    else:
        parent =uniqueData[np.argmax(dataParent[1])]


        #informationGains = [0]*len(features)
        #i=0
        #print(len(features))
        #for feature in features:
            #informationGains[i]=informationGain(dataset,feature,label)
            #i=i+1
        #Here I calculate the information gainf or all features, so I can choose the best
        informationGains = [informationGain(dataset,feature,label) for feature in features]
        #print(informationGains)
        #print(features)
        #The best feature is the one which gives has the maximum information gain
        bestFeatureIndex = np.argmax(informationGains)
        #print(bestFeatureIndex)
        #Now we have the best feature to split our nodes
        bestFeature =features[bestFeatureIndex]
        decisionTree ={bestFeature: {}}
        # ^ it's done in form a of a dict
        features = [i for i in features if i!=bestFeature]
        #as now ^ we have used it in our split, we remove it from and have our remaining features
        for value in np.unique(dataset[bestFeature]):
            divideData = dataset.where(dataset[bestFeature]==value).dropna()
            #This is where aplit occurs with data
            #print(divideData)
            #recursively calling to create subtrees
            subTree =createDecisionTree(divideData,df, features, label, parent)
            #the sub tree is added to the main tree.
            decisionTree[bestFeature][value]= subTree

        #print(decisionTree) to see the trees and subtrees created.
        return(decisionTree)


# In[23]:


def makePredict(decisionTree, testData):
    predictedValue = 0.0
    #print(decisionTree) check all keys for my dictionary
    for key in decisionTree.keys():

        keyValue = testData[key]
        #print(decisionTree[nodes][value])
        #If it can the value for a key, if it doesn't we go to continue
        try:
            decisionTree = decisionTree[key][keyValue]
        except:
            continue;
        #print(decisionTree)

        if type(decisionTree) is dict:
            #print("yes",prediction) this was to check if it is entering the the condition
            #if the type is still dict, we can iterate more to find an answer
            predictedValue = makePredict(decisionTree, testData)
        else:
            predictedValue = decisionTree
            #print("no",prediction)
            return predictedValue
    return predictedValue


# In[24]:


def mostFrequent(List):
    yesCount = 0
    noCount = 0
    for i in range(len(List)):
        if List[i]=='YES':
            yesCount=yesCount+1
        else:
            noCount = noCount+1
    if yesCount > noCount :
        return 'YES'
    else:
        return 'NO'


# In[25]:


def findAccuracy(df, resultFrame, percentage):
    finalValue= ['']*10
    count = 0
    for i in range(len(resultFrame)):
        #print(resultFrame.iloc[i])
        finalValue[i] = mostFrequent(resultFrame.iloc[i])
    #df = pd.read_csv('banks-test.csv')
    #print(finalValue)
    for i in range(len(resultFrame)):
        if finalValue[i]==df['label'][i]:
            count=count+1
    print(count/len(resultFrame)*100)
    convertSeries = pd.Series(finalValue,name="predicted value")
    final_csv = pd.DataFrame()
    final_csv['predicted value' + "  Trees used - " + str(len(resultFrame.columns)) +"  Accuracy - "+ str((count/len(resultFrame))*100) + "% "+ "percentage of Attributes used - " +str((percentage)*100)] = convertSeries
    #print(convertSeries)
    final_csv.to_csv('predicted.csv')


# In[26]:


def forestCreater(df, bankTestData, percentageOfAttributes, numberOfTrees):
    #df = pd.read_csv("banks.csv")
    #print(df)
    #First I remove label as I don't want it to be randomzied and be an option for my random attributes
    noLabelAttributes=len(df.columns)-1
    #Now as the number could be something like 4.5, I take floor to select the number of random attributes
    numberRandomAttributes= math.floor((percentageOfAttributes*noLabelAttributes))
    #This is a data frame where I will be storing my result
    resultFrame = pd.DataFrame()
    for i in range(numberOfTrees):
        #tempFrame is for saving the results for one tree, I need label and parent to send it in my function for initial value.
        #Remove label so I can randomize other attributes
        randomTrainingData= df.iloc[:,:-1]
        tempFrame=['']*10
        label= 'label'
        parent = None
        #print(df)
        randomTrainingData= randomTrainingData.sample(axis=1, n=numberRandomAttributes)
        #print(randomTrainingData) (checking if it works)
        #we make our features according to these attributes
        features = randomTrainingData.columns
        #print(features) (to check)
        #adding labels back as we need it for our tree
        randomTrainingData['label']=df['label']
        #print(randomTrainingData) to check if it is being added back
        #Creating my tree
        decisionTree =createDecisionTree(randomTrainingData, randomTrainingData, features, label, parent)
        #print(decisionTree)
        #now I prepare my testing data , remove labels , and send the transpose to our prediction function
        #bankTestData = pd.read_csv('banks-test.csv')
        bankData = bankTestData.iloc[:,:-1]
        bankData = bankData.T
        #print(len(bankTestData.index)+1)
        # 10 is the number of rows in banks-test
        for j in range(len(bankTestData)):
            #print(len(bankTestData))
            predictedValue = makePredict(decisionTree, bankData[j])
            #print(predictedValue)

            #This is incase some error occurs for some attributes for a test case (I went a little overboard)

            if predictedValue == 0.0:
                designate = randint(0 , 1)
                if designate == 0:
                    predictedValue = 'NO'
                else:
                    predictedValue = 'YES'

            #print(predictedValue)
            #saving it in our temporary list
            tempFrame[j] = predictedValue

        #Now I create a new column each time and get list to be stored in my dataframe
        resultFrame['Prediction ' + str(i+1)] = tempFrame
    #print(resultFrame)
    #Now I find my accuracy using this dataframe.
    findAccuracy(df, resultFrame, percentageOfAttributes)


# In[27]:


def main():
    df = pd.read_csv("banks.csv")
    bankTestData = pd.read_csv('banks-test.csv')
    forestCreater(df, bankTestData, 0.6, 5)

if __name__ == '__main__':
    main()


# In[ ]:

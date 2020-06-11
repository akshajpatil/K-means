
# coding: utf-8

# In[4]:

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:,0] , X[:,1], s=50);
plt.show()


# In[159]:

data=[]
#print(X[299][0])
for i in range(0,300):
    temp=[]
    temp.append(i)
    temp.append(X[i][0])
    temp.append(X[i][1])
    temp.append(X[i][1])
    data.append(temp)
#print(data)


# In[511]:

import pandas as pd  
df = pd.DataFrame(data, columns = ['Name','Column1', 'Column2','Column3'])
#print(len(df))
d1=pd.read_csv("E:/IIIT Delhi/DMG/Assignment 4/clustering-data/animals",sep=" ",header=None)
d2=pd.read_csv("E:/IIIT Delhi/DMG/Assignment 4/clustering-data/countries",sep=" ",header=None)
d3=pd.read_csv("E:/IIIT Delhi/DMG/Assignment 4/clustering-data/fruits",sep=" ",header=None)
d4=pd.read_csv("E:/IIIT Delhi/DMG/Assignment 4/clustering-data/veggies",sep=" ",header=None)
#print(d1.columns)
d1temp=d1.values.tolist()
d2temp=d2.values.tolist()
d3temp=d3.values.tolist()
d4temp=d4.values.tolist()
for i in range (len(d1)):
    d1temp[i][0]=1
    
for i in range (len(d2)):
    d2temp[i][0]=2
for i in range (len(d3)):
    d3temp[i][0]=3
for i in range (len(d4)):
    d4temp[i][0]=4
    
#print(d1.shape)
#d1[0].replace(0,inplace=True)
#print(d1.head())
#data1=d1.values.tolist()+d2.values.tolist()+d3.values.tolist()+d4.values.tolist();
data1=d1temp+d2temp+d3temp+d4temp
#data1=df.values.tolist()
print(len(data1))


# In[519]:

import math
import random
import warnings
warnings.filterwarnings('ignore')
for choice in range(1,4):
    if choice == 1:
        print("Calculated With Euclidean Distance")
    elif choice == 2:
        print("Calculated With Manhattan Distance")
    else:
        print("Calculated With cosine similarity")
    pre=[]
    rec=[]
    fsc=[]
    #choice=0
    for k in range(1,11):
        centi=[]
        for i in range(0,k):
            centi.append(data1[random.randrange(0,len(data1))])
            #print(random.randrange(0,len(df)))

        #print("centi")
        mean=[]

        while(1):
            #k=4
            cluster=[[] for i in range(k)]
            #for i in range(4):
             #   cluster.append([i])

            if choice == 1:
                
                for x in range(len(data1)):
                    evaluateEcud(data1[x])
                    #manhattanDist(data1[x])

                mean=calculateMean(cluster,k)
            elif choice == 2:
                
                for x in range(len(data1)):
                    #evaluateEcud(data1[x])
                    manhattanDist(data1[x])

                mean=calculateMedian(cluster,k)

            else:
                
                for x in range(len(data1)):
                    cosineDist(data1[x])
                    #manhattanDist(data1[x])

                mean=calculateMean(cluster,k)

            #print(len(cluster[0])," ",len(cluster[1])," ",len(cluster[2])," ",len(cluster[3]))

           # print("mean")
            if mean==centi:
                break
            else:
                centi=mean

    #     for w in range(0,k):
    #         print(len(cluster[w]))
    #     print("==========")
        # print(len(cluster[0]))
        # print(len(cluster[1]))
        # print(len(cluster[2]))
        # print(len(cluster[3]))
        count=[]
        for w in range(4):
            count.append(0)
        list_count1=[]   
        for w in range(0,k):
            for s in range(4):
                count[s]=0
            for t in range(0,len(cluster[w])):
                if cluster[w][t][0]==1:
                    count[0]=count[0]+1
                elif cluster[w][t][0]==2:
                    count[1]=count[1]+1
                elif cluster[w][t][0]==3:
                    count[2]=count[2]+1
                else:
                    count[3]=count[3]+1

            temp=[]
            for t23 in range(len(count)):
                temp.append(count[t23])
            list_count1.append(temp)

        print(list_count1,"\n=======")
            #list_count1.append(count)
        pretemp=[]
        rectemp=[]
        fsctemp=[]
        pretemp,rectemp,fsctemp=preRec(list_count1)
        pre.append(pretemp)
        rec.append(rectemp)
        fsc.append(fsctemp)

    #print(pre,"\t",rec,"\t",fsc)
    #print(list_count1,"\n====")
    #print("end==========")

    import matplotlib.pyplot as plt
    x=[1,2,3,4,5,6,7,8,9,10]


    plt.plot(x, pre,label="precision")
    plt.plot(x, rec,label="recall")
    plt.plot(x, fsc,label="fscore")
    plt.legend()
    plt.show()


# In[512]:

from sklearn import preprocessing
# df1temp=pd.DataFrame(d1temp)
# df2temp=pd.DataFrame(d2temp)
# df3temp=pd.DataFrame(d3temp)
# df4temp=pd.DataFrame(d4temp)

df1temp = preprocessing.normalize(d1temp,norm="l2")
df2temp = preprocessing.normalize(d2temp,norm="l2")
df3temp = preprocessing.normalize(d3temp,norm="l2")
df4temp = preprocessing.normalize(d4temp,norm="l2")

d1temp=df1temp.tolist()
d2temp=df2temp.tolist()
d3temp=df3temp.tolist()
d4temp=df4temp.tolist()
# min_max_scaler = preprocessing.MinMaxScaler()

# x_scaled1 = min_max_scaler.fit_transform(df1temp)
# x_scaled2 = min_max_scaler.fit_transform(df2temp)
# x_scaled3 = min_max_scaler.fit_transform(df3temp)
# x_scaled4 = min_max_scaler.fit_transform(df4temp)

# df_normalized1 = pd.DataFrame(x_scaled1)
# df_normalized2 = pd.DataFrame(x_scaled2)
# df_normalized3 = pd.DataFrame(x_scaled3)
# df_normalized4 = pd.DataFrame(x_scaled4)

# d1temp=df_normalized1.values.tolist()
# d2temp=df_normalized2.values.tolist()
# d3temp=df_normalized3.values.tolist()
# d4temp=df_normalized4.values.tolist()

for i in range (len(d1)):
    d1temp[i][0]=1
    
for i in range (len(d2)):
    d2temp[i][0]=2
for i in range (len(d3)):
    d3temp[i][0]=3
for i in range (len(d4)):
    d4temp[i][0]=4
    
#print(d1temp[20])
#d1[0].replace(0,inplace=True)
#print(d1.head())
#data1=d1.values.tolist()+d2.values.tolist()+d3.values.tolist()+d4.values.tolist();
data1=d1temp+d2temp+d3temp+d4temp
#data1=df.values.tolist()
print(len(data1))

import math
import random
import warnings
warnings.filterwarnings('ignore')

pre=[]
rec=[]
fsc=[]
#choice=0
for k in range(1,11):
    centi=[]
    for i in range(0,k):
        centi.append(data1[random.randrange(0,len(data1))])
        #print(random.randrange(0,len(df)))

    #print("centi")
    mean=[]

    while(1):
        #k=4
        cluster=[[] for i in range(k)]
        #for i in range(4):
         #   cluster.append([i])

        

        for x in range(len(data1)):
            evaluateEcud(data1[x])
            #manhattanDist(data1[x])

        mean=calculateMean(cluster,k)
       

        #print(len(cluster[0])," ",len(cluster[1])," ",len(cluster[2])," ",len(cluster[3]))

       # print("mean")
        if mean==centi:
            break
        else:
            centi=mean

#     for w in range(0,k):
#         print(len(cluster[w]))
#     print("==========")
    # print(len(cluster[0]))
    # print(len(cluster[1]))
    # print(len(cluster[2]))
    # print(len(cluster[3]))
    count=[]
    for w in range(4):
        count.append(0)
    list_count1=[]   
    for w in range(0,k):
        for s in range(4):
            count[s]=0
        for t in range(0,len(cluster[w])):
            if cluster[w][t][0]==1:
                count[0]=count[0]+1
            elif cluster[w][t][0]==2:
                count[1]=count[1]+1
            elif cluster[w][t][0]==3:
                count[2]=count[2]+1
            else:
                count[3]=count[3]+1

        temp=[]
        for t23 in range(len(count)):
            temp.append(count[t23])
        list_count1.append(temp)

    print(list_count1,"\n=======")
        #list_count1.append(count)
    pretemp=[]
    rectemp=[]
    fsctemp=[]
    pretemp,rectemp,fsctemp=preRec(list_count1)
    pre.append(pretemp)
    rec.append(rectemp)
    fsc.append(fsctemp)

#print(pre,"\t",rec,"\t",fsc)
#print(list_count1,"\n====")
#print("end==========")

import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7,8,9,10]


plt.plot(x, pre,label="precision")
plt.plot(x, rec,label="recall")
plt.plot(x, fsc,label="fscore")
plt.legend()
plt.show()


# In[107]:

def evaluateEcud(data1):
    
    min=9999999
    j=0
    clus=0
    Sum=0
    for items in centi:
        for i in range(1,len(items)):
            Sum += math.pow(items[i]-data1[i], 2);

        if Sum < min:
            min=Sum
            clus=j
        Sum=math.sqrt(Sum)    
        j=j+1
        #print(Sum)
        Sum=0
    #print(clus)
    cluster[clus].append(data1)
    #print(cluster[clus])


# In[381]:

def manhattanDist(data1):
    
    min=9999999
    j=0
    clus=0
    Sum=0
    for items in centi:
        for i in range(1,len(items)):
            Sum += abs(items[i]-data1[i]);

        if Sum < min:
            min=Sum
            clus=j
        #Sum=math.sqrt(Sum)    
        j=j+1
        #print(Sum)
        Sum=0
    #print(clus)
    cluster[clus].append(data1)
    #print(cluster[clus])


# In[461]:

from sklearn.metrics.pairwise import cosine_similarity
def cosineDist(data1):
    
    min=0.0
    j=0
    clus=0
    Sum=0
    
    cos_lib = cosine_similarity(centi, data1)

    for items in cos_lib:
        if items[0] > min:
            clus=j
            min=items[0]
        j=j+1
    #print(clus)
    cluster[clus].append(data1)
    #print(cluster[clus])


# In[229]:

import numpy as np
def calculateMean(cluster,k):
    mean=[]
    for i in range(k):
        #npArray = np.array(cluster[i])
        mean.append([float(sum(col))/len(col) for col in zip(*cluster[i])])
        #mean.append(np.mean(npArray[:,1:], axis = 0))

    return mean

    


# In[382]:

def calculateMedian(cluster,k):
    median=[]
    for i in range(k):
        dfmedian = pd.DataFrame(cluster[i], columns=None)
        median.append(list(dfmedian.median(axis=0).values))
       # median.append(cluster[i][int(len(cluster[i])/2)])
    #print(median)
    return median


# In[300]:

def preRec(list_count):
    total=(329*328)/2
    tp_fp=0
    tp=0
    fp=0
    fn=0
    tn=0
    #print("list_count",list_count)
    for lists in list_count:
        tp_fp=tp_fp+(sum(lists)*(sum(lists)-1))/2
    #print(tp_fp)
    for lists in list_count:
        for item in lists:
            tp=tp+(item*(item-1))/2
    
    fp=tp_fp-tp
    
    temp1=[]
    temp2=[]
    temp3=[]
    temp4=[]


    for lists in list_count:
        temp1.append(lists[0])
        temp2.append(lists[1])
        temp3.append(lists[2])
        temp4.append(lists[3])
        
    for i in range (len(temp1)):
        for j in range(i+1,len(temp1)):
            fn=fn+temp1[i]*temp1[j]
    
    for i in range (len(temp2)):
        for j in range(i+1,len(temp2)):
            fn=fn+temp2[i]*temp2[j]
    
    for i in range (len(temp3)):
        for j in range(i+1,len(temp3)):
            fn=fn+temp3[i]*temp3[j]
            
    for i in range (len(temp4)):
        for j in range(i+1,len(temp4)):
            fn=fn+temp4[i]*temp4[j]
    
    tn=total-tp-fp-fn
   # print(temp1,"\n",temp2,"\n",temp3,"\n",temp4)
#     print("total=",total)
#     print("tp_fp=",tp_fp)
#     print("tp=",tp)
#     print("fp=",fp)
#     print("fn=",fn)
#     print("tn=",tn)
    
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    fscore=(2*precision*recall)/(precision+recall)
    #print("precision=",precision,"\nrecall=",recall,"\nfscore=",fscore)
    return precision,recall,fscore
    


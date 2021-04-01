# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:28:53 2021

@author: MaryamHashemi
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn
import itertools
#from tensorflow_addons.optimizers import CyclicalLearningRate
import matplotlib as mpl
mpl.style.use('seaborn')
from sklearn.model_selection import train_test_split
import datetime
import time
import pandas as pd

mitbih_train1 = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor-27minutes.csv", dtype={'Time':str})
mitbih_train2 = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor1104-6e22.csv", dtype={'Time':str})
mitbih_train3 = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor1107-6e22.csv", dtype={'Time':str})
mitbih_train4 = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor1111-1ccf.csv", dtype={'Time':str})


print(mitbih_train2.shape)
print (mitbih_train2.dtypes)
print(mitbih_train3.shape)
print (mitbih_train3.dtypes)

newmitbih_train1= pd.DataFrame(mitbih_train1, columns= ['Time','Value'])
newmitbih_train1= mitbih_train1.astype({'Value': int})
mitbih_list1=newmitbih_train1.values.tolist()

newmitbih_train2= pd.DataFrame(mitbih_train2, columns= ['Time','Value'])
newmitbih_train2= mitbih_train2.astype({'Value': int})
mitbih_list2=newmitbih_train2.values.tolist()

newmitbih_train3= pd.DataFrame(mitbih_train3, columns= ['Time','Value'])
newmitbih_train3= mitbih_train3.astype({'Value': int})
mitbih_list3=newmitbih_train3.values.tolist()

newmitbih_train4= pd.DataFrame(mitbih_train4, columns= ['Time','Value'])
newmitbih_train4= mitbih_train4.astype({'Value': int})
mitbih_list4=newmitbih_train4.values.tolist()



data1 = pd.read_excel ("C:/Users/MaryamHashemi/Desktop/ECG/label-27minutes.xlsx",dtype={'Raw Time':str}) 
data2 = pd.read_excel ("C:/Users/MaryamHashemi/Desktop/ECG/label1104-6e22.xlsx",dtype={'Time':str}) 
data3 = pd.read_excel ("C:/Users/MaryamHashemi/Desktop/ECG/label1107-6e22.xlsx",dtype={'Time':str}) 
data4 = pd.read_excel ("C:/Users/MaryamHashemi/Desktop/ECG/label1111-1ccf.xlsx",dtype={'Time':str}) 


ptbdb_labels1 = pd.DataFrame(data1, columns= ['Raw Time','S','D','H','R','TimeStamp (mS)','Calibrated time'])
newptbdb_labels1= ptbdb_labels1.astype({'S': 'int32'})
print (ptbdb_labels1.dtypes)
ptbdb_list1=ptbdb_labels1.values.tolist()

ptbdb_labels2 = pd.DataFrame(data2, columns= ['Time','S','D','H','R','TimeStamp (mS)'])
newptbdb_labels2= ptbdb_labels2.astype({'S': 'int32'})
print (ptbdb_labels2.dtypes)
ptbdb_list2=ptbdb_labels2.values.tolist()

ptbdb_labels3 = pd.DataFrame(data3, columns= ['Time','S','D','H','R','TimeStamp (mS)'])
newptbdb_labels3= ptbdb_labels3.astype({'S': 'int32'})
print (ptbdb_labels3.dtypes)
ptbdb_list3=ptbdb_labels3.values.tolist()

ptbdb_labels4 = pd.DataFrame(data4, columns= ['Time','S','D','H','R','TimeStamp (mS)'])
newptbdb_labels4= ptbdb_labels4.astype({'S': 'int32'})
print (ptbdb_labels4.dtypes)
ptbdb_list4=ptbdb_labels4.values.tolist()



# In[2]: 
import math

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    #s=math.floor(int(s))
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_sec2(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    s1,s2=s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s1)



value1=np.zeros((1,1620,100))
time_list1=[]
for i in range(len(mitbih_list1)-1):
    b=mitbih_list1[i]
    timeval1=get_sec(b[0])
    time_list1.append(timeval1)



value2=np.zeros((1,1644,100))
time_list2=[]
for i in range(len(mitbih_list2)-1):
    b=mitbih_list2[i]
    timeval2=get_sec(b[0])
    time_list2.append(timeval2)
    
    
value3=np.zeros((1,1585,100))
time_list3=[]
for i in range(len(mitbih_list3)-1):
    b=mitbih_list3[i]
    timeval3=get_sec(b[0])
    time_list3.append(timeval3)
    

value4=np.zeros((1,3557,100))
time_list4=[]
for i in range(len(mitbih_list4)-1):
    b=mitbih_list4[i]
    timeval4=get_sec(b[0])
    time_list4.append(timeval4)
    




number=0
array=0
timeval1=[]
for i in range(len(time_list1)):
    if i<(len(time_list1)-1) and time_list1[i]==time_list1[i+1]:
        b=mitbih_list1[i]
        value1[0,number,array]=b[1]
        array=array+1
    else:
        timeval1.append(time_list1[i])
        number=number+1
        array=0
    


number=0
array=0
timeval2=[]
for i in range(len(time_list2)):
    if i<(len(time_list2)-1) and time_list2[i]==time_list2[i+1]:
        b=mitbih_list2[i]
        value2[0,number,array]=b[1]
        array=array+1
    else:
        timeval2.append(time_list2[i])
        number=number+1
        array=0
        
        

number=0
array=0
timeval3=[]
for i in range(len(time_list3)):
    if i<(len(time_list3)-1) and time_list3[i]==time_list3[i+1]:
        b=mitbih_list3[i]
        value3[0,number,array]=b[1]
        array=array+1
    else:
        timeval3.append(time_list3[i])
        number=number+1
        array=0
        
        
number=0
array=0
timeval4=[]
for i in range(0,355598):
    if i<(len(time_list4)-1) and time_list4[i+3]==time_list4[i+4]:
        b=mitbih_list4[i+3]
        value4[0,number,array]=b[1]
        array=array+1
    else:
        timeval4.append(time_list4[i+3])
        number=number+1
        array=0

# In[3]: 
    
    
time2_list1=[]
S_list1=[]
D_list1=[]
H_list1=[]
R_list1=[]

for i in range(len(ptbdb_list1)):
    if i==1241 or i==3225 or i==3224 or i==3508 or i==3509 or i==3584 or i==3585 or i==5546:
        b=ptbdb_list1[i]
        time=get_sec(b[0])
        time2_list1.append(time)
        S_list1.append(b[1])
        D_list1.append(b[2])
        H_list1.append(b[3])
        R_list1.append(b[4])
    else: 
        b=ptbdb_list1[i]
        time=get_sec2(b[0])
        time2_list1.append(time)
        S_list1.append(b[1])
        D_list1.append(b[2])
        H_list1.append(b[3])
        R_list1.append(b[4])
        
        
        
        
time2_list2=[]
S_list2=[]
D_list2=[]
H_list2=[]
R_list2=[]

for i in range(len(ptbdb_list2)):    
    if  i==1976 or i==0 or i==1 or i==1974 or i==1975 :
        b=ptbdb_list2[i]
        time=get_sec(b[0])
        time2_list2.append(time)
        S_list2.append(b[1])
        D_list2.append(b[2])
        H_list2.append(b[3])
        R_list2.append(b[4])
         
        
    else:
        b=ptbdb_list2[i]
        time=get_sec2(b[0])
        time2_list2.append(time)
        S_list2.append(b[1])
        D_list2.append(b[2])
        H_list2.append(b[3])
        R_list2.append(b[4])
        
        
        

     
time2_list3=[]
S_list3=[]
D_list3=[]
H_list3=[]
R_list3=[]

for i in range(len(ptbdb_list3)):    
    if  i==2711 or i==2712  :
        b=ptbdb_list3[i]
        time=get_sec(b[0])
        time2_list3.append(time)
        S_list3.append(b[1])
        D_list3.append(b[2])
        H_list3.append(b[3])
        R_list3.append(b[4])
         
    elif i==0:
        time2_list3.append(48188)
        S_list3.append(117)
        D_list3.append(72)
        H_list3.append(0)
        R_list3.append(0)
        
    else:
        b=ptbdb_list3[i]
        time=get_sec2(b[0])
        time2_list3.append(time)
        S_list3.append(b[1])
        D_list3.append(b[2])
        H_list3.append(b[3])
        R_list3.append(b[4])



time2_list4=[]
S_list4=[]
D_list4=[]
H_list4=[]
R_list4=[]

for i in range(len(ptbdb_list4)):    
    if  i==4283 or i==5973 or i==5974:
        b=ptbdb_list4[i]
        time=get_sec(b[0])
        time2_list4.append(time)
        S_list4.append(b[1])
        D_list4.append(b[2])
        H_list4.append(b[3])
        R_list4.append(b[4])
         
    elif i==0:
        time2_list4.append(44002)
        S_list4.append(110)
        D_list4.append(52)
        H_list4.append(0)
        R_list4.append(0)
        
    else:
        b=ptbdb_list4[i]
        time=get_sec2(b[0])
        time2_list4.append(time)
        S_list4.append(b[1])
        D_list4.append(b[2])
        H_list4.append(b[3])
        R_list4.append(b[4])






# In[4]: 


nS_list1=[]
nD_list1=[]
nH_list1=[]
nR_list1=[]
smoothtime1=[]
count=0
for i in range(len(time2_list1)):
    if count<len(S_list1)-3: 

        if time2_list1[count]==time2_list1[count+1]:
            if time2_list1[count+1]==time2_list1[count+2]:
                nS_list1.append((S_list1[count]+S_list1[count+1]+S_list1[count+2])/3)
                nD_list1.append((D_list1[count]+D_list1[count+1]+D_list1[count+2])/3)
                nH_list1.append((H_list1[count]+H_list1[count+1]+H_list1[count+2])/3)
                nR_list1.append((R_list1[count]+R_list1[count+1]+R_list1[count+2])/3)
                smoothtime1.append(time2_list1[count])
                count=count+3
                
            else:
                nS_list1.append((S_list1[count]+S_list1[count+1])/2)
                nD_list1.append((D_list1[count]+D_list1[count+1])/2)
                nH_list1.append((H_list1[count]+H_list1[count+1])/2)
                nR_list1.append((R_list1[count]+R_list1[count+1])/2)
                smoothtime1.append(time2_list1[count])
                count=count+2
            
        else:
            nS_list1.append(S_list1[count])
            nD_list1.append(D_list1[count])
            nH_list1.append(H_list1[count])
            nR_list1.append(R_list1[count])
            smoothtime1.append(time2_list1[count])
            count=count+1
        
    else:
        break
    
    
    
    
    
nS_list2=[]
nD_list2=[]
nH_list2=[]
nR_list2=[]
smoothtime2=[]
count=0
for i in range(len(time2_list2)):
    if count<len(S_list2)-3: 

        if time2_list2[count]==time2_list2[count+1]:
            if time2_list2[count+1]==time2_list2[count+2]:
                nS_list2.append((S_list2[count]+S_list2[count+1]+S_list2[count+2])/3)
                nD_list2.append((D_list2[count]+D_list2[count+1]+D_list2[count+2])/3)
                nH_list2.append((H_list2[count]+H_list2[count+1]+H_list2[count+2])/3)
                nR_list2.append((R_list2[count]+R_list2[count+1]+R_list2[count+2])/3)
                smoothtime2.append(time2_list2[count])
                count=count+3
                
            else:
                nS_list2.append((S_list2[count]+S_list2[count+1])/2)
                nD_list2.append((D_list2[count]+D_list2[count+1])/2)
                nH_list2.append((H_list2[count]+H_list2[count+1])/2)
                nR_list2.append((R_list2[count]+R_list2[count+1])/2)
                smoothtime2.append(time2_list2[count])
                count=count+2
            
        else:
            nS_list2.append(S_list2[count])
            nD_list2.append(D_list2[count])
            nH_list2.append(H_list2[count])
            nR_list2.append(R_list2[count])
            smoothtime2.append(time2_list2[count])
            count=count+1
        
    else:
        break





nS_list3=[]
nD_list3=[]
nH_list3=[]
nR_list3=[]
smoothtime3=[]
count=0
for i in range(len(time2_list3)):
    if count<len(S_list3)-3: 

        if time2_list3[count]==time2_list3[count+1]:
            if time2_list3[count+1]==time2_list3[count+2]:
                nS_list3.append((S_list3[count]+S_list3[count+1]+S_list3[count+2])/3)
                nD_list3.append((D_list3[count]+D_list3[count+1]+D_list3[count+2])/3)
                nH_list3.append((H_list3[count]+H_list3[count+1]+H_list3[count+2])/3)
                nR_list3.append((R_list3[count]+R_list3[count+1]+R_list3[count+2])/3)
                smoothtime3.append(time2_list3[count])
                count=count+3
                
            else:
                nS_list3.append((S_list3[count]+S_list3[count+1])/2)
                nD_list3.append((D_list3[count]+D_list3[count+1])/2)
                nH_list3.append((H_list3[count]+H_list3[count+1])/2)
                nR_list3.append((R_list3[count]+R_list3[count+1])/2)
                smoothtime3.append(time2_list3[count])
                count=count+2
            
        else:
            nS_list3.append(S_list3[count])
            nD_list3.append(D_list3[count])
            nH_list3.append(H_list3[count])
            nR_list3.append(R_list3[count])
            smoothtime3.append(time2_list3[count])
            count=count+1
        
    else:
        break
    
    
    
    
    
nS_list4=[]
nD_list4=[]
nH_list4=[]
nR_list4=[]
smoothtime4=[]
count=0
for i in range(len(time2_list4)):
    if count<len(S_list4)-3: 

        if time2_list4[count]==time2_list4[count+1]:
            if time2_list4[count+1]==time2_list4[count+2]:
                nS_list4.append((S_list4[count]+S_list4[count+1]+S_list4[count+2])/3)
                nD_list4.append((D_list4[count]+D_list4[count+1]+D_list4[count+2])/3)
                nH_list4.append((H_list4[count]+H_list4[count+1]+H_list4[count+2])/3)
                nR_list4.append((R_list4[count]+R_list4[count+1]+R_list4[count+2])/3)
                smoothtime4.append(time2_list4[count])
                count=count+3
                
            else:
                nS_list4.append((S_list4[count]+S_list4[count+1])/2)
                nD_list4.append((D_list4[count]+D_list4[count+1])/2)
                nH_list4.append((H_list4[count]+H_list4[count+1])/2)
                nR_list4.append((R_list4[count]+R_list4[count+1])/2)
                smoothtime4.append(time2_list4[count])
                count=count+2
            
        else:
            nS_list4.append(S_list4[count])
            nD_list4.append(D_list4[count])
            nH_list4.append(H_list4[count])
            nR_list4.append(R_list4[count])
            smoothtime4.append(time2_list4[count])
            count=count+1
        
    else:
        break


#nS_list, nD_list, nH_list, nR_list are our labels. smoothtime is equvalent time. value is our train data and timeval is its equvalent time.


# In[3]:


finaltime1=[]
final_label1=np.zeros((1,1620,4))

smoothtime21=smoothtime1[56:850]
nS_list21=nS_list1[56:850]
nD_list21=nD_list1[56:850]
nH_list21=nH_list1[56:850]
nR_list21=nR_list1[56:850]

for i in range(len(smoothtime21)):
    for j in range(len(timeval1)):
        if smoothtime21[i]==timeval1[j]:
            if final_label1[0,j,0]==0:
                final_label1[0,j,0]=nS_list21[i]
                final_label1[0,j,1]=nD_list21[i]
                final_label1[0,j,2]=nH_list21[i]
                final_label1[0,j,3]=nR_list21[i]
                finaltime1.append(timeval1[j])
        
        elif smoothtime21[i]>timeval1[j]:
            if final_label1[0,j,0]==0:
                final_label1[0,j,0]=nS_list21[i]
                final_label1[0,j,1]=nD_list21[i]
                final_label1[0,j,2]=nH_list21[i]
                final_label1[0,j,3]=nR_list21[i]
                finaltime1.append(timeval1[j])

final_label1[0,1619,0]=94.25
final_label1[0,1619,1]=61.25
final_label1[0,1619,2]=71.75
final_label1[0,1619,3]=62.5          










finaltime2=[]
final_label2=np.zeros((1,1644,4))
smoothtime22=smoothtime2[104:1880]
nS_list22=nS_list2[104:1880]
nD_list22=nD_list2[104:1880]
nH_list22=nH_list2[104:1880]
nR_list22=nR_list2[104:1880]

for i in range(len(smoothtime22)):
    for j in range(len(timeval2)):
        if smoothtime22[i]==timeval2[j]:
            if final_label2[0,j,0]==0:
                final_label2[0,j,0]=nS_list22[i]
                final_label2[0,j,1]=nD_list22[i]
                final_label2[0,j,2]=nH_list22[i]
                final_label2[0,j,3]=nR_list22[i]
                finaltime2.append(timeval2[j])
        
        elif smoothtime22[i]>timeval2[j]:
            if final_label2[0,j,0]==0:
                final_label2[0,j,0]=nS_list22[i]
                final_label2[0,j,1]=nD_list22[i]
                final_label2[0,j,2]=nH_list22[i]
                final_label2[0,j,3]=nR_list22[i]
                finaltime2.append(timeval2[j])




final_label2[0,1619,0]=94.25
final_label2[0,1619,1]=61.25
final_label2[0,1619,2]=71.75
final_label1[0,1619,3]=62.5          

        
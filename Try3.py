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

mitbih_test = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor-27minutes.csv", header=None)
mitbih_train = pd.read_csv("C:/Users/MaryamHashemi/Desktop/ECG/sensor-27minutes.csv", dtype={'Time':str})

print(mitbih_train.shape)
print (mitbih_train.dtypes)

newmitbih_train= pd.DataFrame(mitbih_train, columns= ['Time','Value'])
newmitbih_train= mitbih_train.astype({'Value': int})
mitbih_list=newmitbih_train.values.tolist()


data = pd.read_excel ("C:/Users/MaryamHashemi/Desktop/ECG/label-27minutes.xlsx",dtype={'Raw Time':str}) 
ptbdb_labels = pd.DataFrame(data, columns= ['Raw Time','S','D','H','R','TimeStamp (mS)','Calibrated time'])
newptbdb_labels= ptbdb_labels.astype({'S': 'int32'})
print (ptbdb_labels.dtypes)
ptbdb_list=ptbdb_labels.values.tolist()



# In[2]: 

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_sec2(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    s1,s2=s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s1)



value=np.zeros((1,1620,100))
time_list=[]
for i in range(len(mitbih_list)):
    b=mitbih_list[i]
    timeval=get_sec(b[0])
    time_list.append(timeval)


number=0
array=0
timeval=[]
for i in range(len(time_list)):
    if i<(len(time_list)-1) and time_list[i]==time_list[i+1]:
        b=mitbih_list[i]
        value[0,number,array]=b[1]
        array=array+1
    else:
        timeval.append(time_list[i])
        number=number+1
        array=0
    


# In[3]: 
    
    
time2_list=[]
S_list=[]
D_list=[]
H_list=[]
R_list=[]

for i in range(len(ptbdb_list)):
    if i==1241 or i==3225 or i==3224 or i==3508 or i==3509 or i==3584 or i==3585 or i==5546:
        b=ptbdb_list[i]
        time=get_sec(b[0])
        time2_list.append(time)
        S_list.append(b[1])
        D_list.append(b[2])
        H_list.append(b[3])
        R_list.append(b[4])
    else: 
        b=ptbdb_list[i]
        time=get_sec2(b[0])
        time2_list.append(time)
        S_list.append(b[1])
        D_list.append(b[2])
        H_list.append(b[3])
        R_list.append(b[4])




nS_list=[]
nD_list=[]
nH_list=[]
nR_list=[]
smoothtime=[]
count=-1
for i in range(len(time2_list)-10):
    if count<len(S_list)-3: 
        count=count+1
    else:
        break
    if time2_list[count]==time2_list[count+1]:
        if time2_list[count+1]==time2_list[count+2]:
            nS_list.append((S_list[count]+S_list[count+1]+S_list[count+2])/3)
            nD_list.append((D_list[count]+D_list[count+1]+D_list[count+2])/3)
            nH_list.append((H_list[count]+H_list[count+1]+H_list[count+2])/3)
            nR_list.append((R_list[count]+R_list[count+1]+R_list[count+2])/3)
            smoothtime.append(time2_list[count])
            count=count+2
            
        else:
            nS_list.append((S_list[count]+S_list[count+1])/2)
            nD_list.append((D_list[count]+D_list[count+1])/2)
            nH_list.append((H_list[count]+H_list[count+1])/2)
            nR_list.append((R_list[count]+R_list[count+1])/2)
            smoothtime.append(time2_list[count])
            count=count+1
        
    else:
        nS_list.append(S_list[count])
        nD_list.append(D_list[count])
        nH_list.append(H_list[count])
        nR_list.append(R_list[count])
        smoothtime.append(time2_list[count])
        
#t=0
for i in range(len(smoothtime)):
   # if t<len(timeval): 
     
       # if smoothtime[i]==timeval[t]:
            
           # final_label[0,t,0]=nS_list[i]
           # final_label[0,t,1]=nD_list[i]
           # final_label[0,t,2]=nH_list[i]
           # final_label[0,t,3]=nR_list[i]
           # finaltime.append(timeval[t])
          #  t=t+1
            
       # elif smoothtime[i]>timeval[t]:
           # final_label[0,t,0]=nS_list[i-1]
            #final_label[0,t,1]=nD_list[i-1]
          #  final_label[0,t,2]=nH_list[i-1]
           # final_label[0,t,3]=nR_list[i-1]
          #  finaltime.append(timeval[t])
          #  t=t+1
           # if t==1620:
             #   t=t-1
           # if smoothtime[i]==timeval[t]:
              #  final_label[0,t,0]=nS_list[i]
             #   final_label[0,t,1]=nD_list[i]
             #   final_label[0,t,2]=nH_list[i]
            #    final_label[0,t,3]=nR_list[i]
             #   finaltime.append(timeval[t])
             #   t=t+1 
          #  elif smoothtime[i]>timeval[t]:
             #   final_label[0,t,0]=nS_list[i-1]
           #     final_label[0,t,1]=nD_list[i-1]
           #     final_label[0,t,2]=nH_list[i-1]
            #    final_label[0,t,3]=nR_list[i-1]
            #    finaltime.append(timeval[t])
            #    t=t+1 
            
   # else:
       # break        
   
#nS_list, nD_list, nH_list, nR_list are our labels. smoothtime is equvalent time. value is our train data and timeval is its equvalent time.


# In[3]:



finaltime=[]
final_label=np.zeros((1,1620,4))

t=0
for i in range(len(smoothtime)):
    
    if t<len(timeval)-1: 
     
        if smoothtime[i]==timeval[t]:
            
            final_label[0,t,0]=nS_list[i]
            final_label[0,t,1]=nD_list[i]
            final_label[0,t,2]=nH_list[i]
            final_label[0,t,3]=nR_list[i]
            finaltime.append(timeval[t])
            t=t+1
            
        elif smoothtime[i]>timeval[t]:
            final_label[0,t,0]=nS_list[i-1]
            final_label[0,t,1]=nD_list[i-1]
            final_label[0,t,2]=nH_list[i-1]
            final_label[0,t,3]=nR_list[i-1]
            finaltime.append(timeval[t])
            t=t+1
            if smoothtime[i]==timeval[t]:
                final_label[0,t,0]=nS_list[i]
                final_label[0,t,1]=nD_list[i]
                final_label[0,t,2]=nH_list[i]
                final_label[0,t,3]=nR_list[i]
                finaltime.append(timeval[t])
                t=t+1            
    else:
        break        

# In[4]: 

y_train, y_test,X_train, X_test  = train_test_split(final_label[0,:,0], value[0,:,:], test_size=0.3, shuffle=True)
#y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)            
           

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

# In[5]:

from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adadelta, Adagrad
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.metrics import classification_report

opt = SGD(lr=0.1, momentum=0.9)

#def buildModel(xtrain,ytrain):
batch_size=100
epochs=30
model= Sequential()
model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu', input_shape=(100,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(32,return_sequences=True))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy']) 
model.summary()



history=model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_valid, y_valid))
model.save("toxic.h5")
# pred=model.predict(X_test)



# In[6]:

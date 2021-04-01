
            


# In[4]: 
    
import math  
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(-1,1))

# fit scaler on data
#VAL=scaler.fit(value[0,:,:])
#normalized = scaler.transform(value[0,:,:])
  
maxlabel=max(final_label[0,:,0])
minlabel=min(final_label[0,:,0])


y_train, y_test,X_train, X_test  = train_test_split(final_label[0,:,0], value[0,:,:], test_size=0.3, shuffle=True)
#y_train, y_test,X_train, X_test  = train_test_split(labelclass, labelname, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)            
           

#X_train_reshape=np.zeros((1,(100*len(X_train))))
#n=-1
#for i in range(len(X_train)):
#    for j in range(X_train.shape[1]):
#        n=n+1
#        X_train_reshape[0,n]=X_train[i,j]
    

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)



pre_label_train=np.zeros((len(y_train),math.ceil(maxlabel-minlabel)+1))
pre_label_test=np.zeros((len(y_test),math.ceil(maxlabel-minlabel)+1))
pre_label_valid=np.zeros((len(y_valid),math.ceil(maxlabel-minlabel)+1))


for i in range (len(y_train)):
    count=y_train[i]-81
    count=round(count)
    count=int(count)
    pre_label_train[i,count]=1
    

for i in range (len(y_test)):
    count=y_test[i]-81
    count=round(count)
    count=int(count)
    pre_label_test[i,count]=1
    

for i in range (len(y_valid)):
    count=y_valid[i]-81
    count=round(count)
    count=int(count)
    pre_label_valid[i,count]=1

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
dense=math.ceil(maxlabel-minlabel)+1
#def buildModel(xtrain,ytrain):
batch_size=32
epochs=150
model= Sequential()
model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu', input_shape=(100,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
#model.add(LSTM(64,return_sequences=True))
model.add(LSTM(32,return_sequences=True))
model.add(Flatten())
model.add(Dense(32,activation='softmax'))
model.add(Dropout(0.45))
model.add(Dense(dense,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam' ,metrics=['accuracy']) 
model.summary()



history=model.fit(X_train,pre_label_train,batch_size=batch_size,epochs=epochs,validation_data=(X_valid, pre_label_valid))
#history=model.fit(X_train_reshape,pre_label_train,batch_size=batch_size,epochs=epochs)
model.save("toxic.h5")
# pred=model.predict(X_test)



# In[6]:
    
    
scorevalid = model.evaluate(X_valid, pre_label_valid, batch_size=32)
scoretest = model.evaluate(X_test, pre_label_test, batch_size=32)


#print('Valid score:', scorevalid[0])
#print('Valid accuracy:', scorevalid[1])

#print('test score:', scoretest[0])
print('test accuracy:', scoretest[1])
# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Val Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()


# In[7]:
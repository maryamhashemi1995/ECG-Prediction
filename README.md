# ECG-Prediction
***********************
* Problem  Statement  *
***********************
The folder contains two files of 27 minutes data each:
    |- sensor-27minutes.csv     # the raw sensor data file with sampling rate 100Hz (also at https://snapshot.raintank.io/dashboard/snapshot/m7rUj15wHQwUXrputwY1W3rrVP46ndpO)
    |- label-27minutes.csv      # the corresppinding label of four system parameters Heartrate (H), Respiratoryrate (R), Systolic (S) and Diasytic (D). 
                                # the timestamp of each row is the sum of the 2nd row's time and the corresponding 
                                # TimeStamp difference (in millisecond) between the current row and the 2nd row. 
                                # The example is calculated in column H.


Write a stream data analytics program to build the relationship model between the sensor data and the four parameters (S, D, H, R), 
and use the built model to predict the future four parameters per second from future raw sensor data. Python language 
is recommended for the stream data analytics program.

Please use first 20 minutes data for training, validation and testing. The remaining 7 minutes 
data is used as the new data to calculate/predict each of the four parameters per second and evaluate the MAE (mean absolute error) of 
each parameter per second. 

*************************

1107-6e22: 
https://snapshot.raintank.io/dashboard/snapshot/sF1HKj4AKauQpCAgiJnKRtro699BAhSM

1104-6e22:
https://snapshot.raintank.io/dashboard/snapshot/zR9aPCnpa02WtAwLX77e0VXlg9AEn0Yh

1111-1ccf:
https://snapshot.raintank.io/dashboard/snapshot/eOIKLPzgNgBXxx12aezLEWEwJaPm8Y6T

**************
* Tips       *
**************
1. The stream data analytics program will be a combination of signal processing and machine/deep learning regresssion model;
2. The sensor data is seismocardiogram (SCG) data. Those four parameters are Heartrate (H), Respiratoryrate (R), Systolic (S) and Diasytic (D),
which are related to the features of peaks and envelopes (see SCG.jpg for details);
3. There are some abnorml events/interferences or missing data during some periods (which is common in real-world sensor data), and the beginning 
rows of the label files with R=0 can be discarded, the program shall automatically identify and remove them; 
4. Time series data feature extraction may leverage python open source library tsfresh (https://tsfresh.readthedocs.io/en/latest/).

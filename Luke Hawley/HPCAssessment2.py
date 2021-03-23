import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)
import pickle
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(1000000)

df_1 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS1.zip",header=None)
df_2 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS2.zip",header=None)
df_3 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS3.zip",header=None)
df_4 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS4.zip",header=None)

colnames = (['source_ip', 'source_port', 'dest_ip', 'dest_port', 'proto', 'state', 'duration', 'source_bytes', 'dest_bytes', 'source_ttl',
             'dest_ttl', 'source_loss', 'dest_loss', 'service', 'source_load', 'dest_load', 'source_pkts', 'dest_pkts', 'source_TP_win', 'dest_TP_win', 
             'source_tcp_bn', 'dest_tcp_bn', 'source_mean_sz', 'dest_mean_sz', 'trans_depth', 'res_bdy_len', 'source_jitter', 'dest_jitter', 'start_time',
             'last_time', 'source_int_pk_time', 'dest_int_pk_time', 'tcp_rtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'count_state_ttl', 
             'count_flw_http_mthd', 'is_ftp_login', 'count_ftp_cmd', 'count_srv_source', 'count_srv_dest', 'count_dest_ltm',
             'count_source_ltm', 'count_source_destport_ltm', 'count_dest_sourceport_ltm', 'counts_dest_source_ltm', 'attack_cat', 'Label'])
df_1.columns = colnames
df_2.columns = colnames
df_3.columns = colnames
df_4.columns = colnames

data_frames = [df_1,df_2,df_3,df_4]
data = pd.concat(data_frames)
data =data.fillna(0)
data.groupby('count_flw_http_mthd').size()
data.groupby('is_ftp_login').size()
data.groupby('count_ftp_cmd').size()
data['count_ftp_cmd'] = data['count_ftp_cmd'].replace(' ',0)
data.groupby('count_ftp_cmd').size()



data_source_ip = pd.DataFrame(data['source_ip'])
data_source_port = pd.DataFrame(data['source_port'])
data_dest_ip = pd.DataFrame(data['dest_ip'])
data_dest_port = pd.DataFrame(data['dest_port'])
data_proto = pd.DataFrame(data['proto'])
data_state = pd.DataFrame(data['state'])
data_service = pd.DataFrame(data['service'])
data_count_ftp_cmd = pd.DataFrame(data['count_ftp_cmd'])
data_attack_cat = pd.DataFrame(data['attack_cat'])

df['attack_cat'] = df['attack_cat'].map({'Normal': 'Normal', 'Exploits': 'Exploits', ' Fuzzers ': 'Fuzzers', 'DoS': 'DoS',
                                          ' Reconnaissance ': 'Reconnaissance', ' Fuzzers': 'Fuzzers', 'Analysis': 'Analysis',
                                         'Backdoor': 'Backdoor', 'Reconnaissance': 'Reconnaissance',  ' Shellcode ': 'Shellcode',
                                         'Backdoors': 'Backdoor', 'Shellcode': 'Shellcode',  'Worms': 'Worms', 'Generic': 'Generic'})

sips = data.source_ip.unique()
sip_dict = dict(zip(sips,range(len(sips))))

sp = data.source_port.unique()
sp_dict = dict(zip(sp,range(len(sp))))
               
dips = data.dest_ip.unique()
dip_dict = dict(zip(dips,range(len(dips))))

dp = data.dest_port.unique()
dp_dict = dict(zip(dp,range(len(dp))))

p = data.proto.unique()
p_dict = dict(zip(p,range(len(p))))

states = data.state.unique()
state_dict = dict(zip(states,range(len(states))))

services = data.service.unique()
service_dict = dict(zip(services,range(len(services))))

cfc = data.count_ftp_cmd.unique()
cfc_dict = dict(zip(cfc,range(len(cfc))))

ac = data.attack_cat.unique()
ac_dict = dict(zip(ac,range(len(ac))))

data['source_ip_int'] = data['source_ip'].map(sip_dict)
data['source_port_int'] = data['source_port'].map(sp_dict)
data['dest_ip_int'] = data['dest_ip'].map(dip_dict)
data['dest_port_int'] = data['dest_port'].map(dp_dict)
data['proto_int'] = data['proto'].map(p_dict)
data['state_int'] = data['state'].map(state_dict)
data['service_int'] = data['service'].map(service_dict)
data['count_ftp_cmd_int'] = data['count_ftp_cmd'].map(cfc_dict)
data['attack_cat_int'] = data['attack_cat'].map(ac_dict)

data = data.drop('source_ip',axis=1)
data = data.drop('source_port',axis=1)
data = data.drop('dest_ip',axis=1)
data = data.drop('dest_port',axis=1)
data = data.drop('proto',axis=1)
data = data.drop('state',axis=1)
data = data.drop('service',axis=1)
data = data.drop('count_ftp_cmd',axis=1)
data = data.drop('attack_cat',axis=1)

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)
data_scaled.shape

from sklearn.model_selection import train_test_split
import datetime as dt


Y = data['attack_cat_int']
X_train, X_test, Y_train, Y_test = train_test_split(data_scaled, Y, test_size = 0.1, random_state = 10)

import keras
from keras import layers, models
from keras.layers import Dense

encoding_dim = 128

input_data = keras.Input(shape=X_train.shape[1])
#Encoded representation
encoded = layers.Dense(encoding_dim, activation='relu')(input_data)
#Adding layers
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
#Loss reconstruction
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(X_train.shape[1], activation='softmax')(decoded)
#Maps the input to its reconstruction
autoencoder = keras.Model(input_data, decoded)

def compile_fit(model, max_epochs, step_size):
    
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    epochs_list=list()
    trainacc=list()
    testacc=list()
    
    epochs = 0
    
    for i in range(0, int(max_epochs / step_size)):
        
        print("epoch : " + str(epochs))
        
        model.fit(X_train, Y_train, epochs=step_size, batch_size=256, validation_data=(X_test, Y_test), 
          callbacks=[tensorboard_callback])
        trainscores = model.evaluate(X_train, Y_train)
        testscores = model.evaluate(X_test, Y_test)
        
        trainacc.append(trainscores[1])
        testacc.append(testscores[1])
        epochs = epochs + step_size
        epochs_list.append(epochs)
        
    return epochs_list, trainacc, testacc



epochs, trainacc, testacc = compile_fit(autoencoder, 1000, 2)


trainscores = autoencoder.evaluate(X_train, Y_train)
testscores = autoencoder.evaluate(X_test, Y_test)

autoencoder.save('AutoEncodeHPC')
pickle.dump(trainacc, open('trainacc_auto.pkl','wb'))
pickle.dump(testacc, open('testacc_auto.pkl','wb'))

predictions = autoencoder.predict(X_test)
Y_pred = predictions.argmax(axis=1)
Y_test_c = Y_test.values.tolist()

pickle.dump(Y_pred, open('Y_predictions_auto.pkl','wb'))
pickle.dump(Y_test_c, open('Y_test_c_auto.pkl','wb'))

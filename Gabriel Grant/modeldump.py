import pickle
import tensorflow as tf
import gzip
import sys
import sklearn
import datetime as dt
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(1000000)

start = dt.datetime.now()
print("Reading df1")
df1 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS1.zip",header=None)
print("Reading df2")
df2 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS2.zip",header=None)
print("Reading df3")
df3 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS3.zip",header=None)
print("Reading df4")
df4 = pd.read_csv("https://github.com/Galeforse/DST-Assessment-04/raw/main/Data/UNS4.zip",header=None)
print("Data fetched in:" ,dt.datetime.now()-start)

df = pd.concat([df1,df2,df3,df4],ignore_index=True)
df.columns = ['source_ip', 'source_port', 'dest_ip', 'dest_port', 'proto', 'state', 'duration', 'source_bytes', 'dest_bytes', 'source_ttl',
             'dest_ttl', 'source_loss', 'dest_loss', 'service', 'source_load', 'dest_load', 'source_pkts', 'dest_pkts', 'source_TP_win', 'dest_TP_win', 
             'source_tcp_bn', 'dest_tcp_bn', 'source_mean_sz', 'dest_mean_sz', 'trans_depth', 'res_bdy_len', 'source_jitter', 'dest_jitter', 'start_time',
             'last_time', 'source_int_pk_time', 'dest_int_pk_time', 'tcp_rtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'count_state_ttl', 
             'count_flw_http_mthd', 'is_ftp_login', 'count_ftp_cmd', 'count_srv_source', 'count_srv_dest', 'count_dest_ltm',
             'count_source_ltm', 'count_source_destport_ltm', 'count_dest_sourceport_ltm', 'counts_dest_source_ltm', 'attack_cat', 'Label']

df['attack_cat'] = df['attack_cat'].fillna('Normal')

attackcount = pd.DataFrame(df['attack_cat'].value_counts())

ac = []
for i in attackcount.index:
    ac.append(i)
ac
an = attackcount["attack_cat"].tolist()
df = df.fillna(0)
df['count_ftp_cmd'] = df['count_ftp_cmd'].apply(pd.to_numeric,errors="coerce")
df = df.fillna(0)

df['attack_cat'] = df['attack_cat'].map({'Normal': 'Normal', 'Exploits': 'Exploits', ' Fuzzers ': 'Fuzzers', 'DoS': 'DoS',
                                          ' Reconnaissance ': 'Reconnaissance', ' Fuzzers': 'Fuzzers', 'Analysis': 'Analysis',
                                         'Backdoor': 'Backdoor', 'Reconnaissance': 'Reconnaissance',  ' Shellcode ': 'Shellcode',
                                         'Backdoors': 'Backdoor', 'Shellcode': 'Shellcode',  'Worms': 'Worms', 'Generic': 'Generic'})
df.groupby('attack_cat').size()

df = df.drop('Label',axis=1)
df_source_ip = pd.DataFrame(df['source_ip'])
df_source_port = pd.DataFrame(df['source_port'])
df_dest_ip = pd.DataFrame(df['dest_ip'])
df_dest_port = pd.DataFrame(df['dest_port'])
df_proto = pd.DataFrame(df['proto'])
df_state = pd.DataFrame(df['state'])
df_service = pd.DataFrame(df['service'])
df_count_ftp_cmd = pd.DataFrame(df['count_ftp_cmd'])
df_attack_cat = pd.DataFrame(df['attack_cat'])

print("creating dictionaries")
# we now create dictionaries to allow us to map onto the data frame

sips = df.source_ip.unique()
sip_dict = dict(zip(sips,range(len(sips))))

sp = df.source_port.unique()
sp_dict = dict(zip(sp,range(len(sp))))
               
dips = df.dest_ip.unique()
dip_dict = dict(zip(dips,range(len(dips))))

dp = df.dest_port.unique()
dp_dict = dict(zip(dp,range(len(dp))))

p = df.proto.unique()
p_dict = dict(zip(p,range(len(p))))

states = df.state.unique()
state_dict = dict(zip(states,range(len(states))))

services = df.service.unique()
service_dict = dict(zip(services,range(len(services))))

cfc = df.count_ftp_cmd.unique()
cfc_dict = dict(zip(cfc,range(len(cfc))))

ac = df.attack_cat.unique()
ac_dict = dict(zip(ac,range(len(ac))))

df['source_ip_int'] = df['source_ip'].map(sip_dict)
df['source_port_int'] = df['source_port'].map(sp_dict)
df['dest_ip_int'] = df['dest_ip'].map(dip_dict)
df['dest_port_int'] = df['dest_port'].map(dp_dict)
df['proto_int'] = df['proto'].map(p_dict)
df['state_int'] = df['state'].map(state_dict)
df['service_int'] = df['service'].map(service_dict)
df['count_ftp_cmd_int'] = df['count_ftp_cmd'].map(cfc_dict)
df['attack_cat_int'] = df['attack_cat'].map(ac_dict)

df = df.drop('source_ip',axis=1)
df = df.drop('source_port',axis=1)
df = df.drop('dest_ip',axis=1)
df = df.drop('dest_port',axis=1)
df = df.drop('proto',axis=1)
df = df.drop('state',axis=1)
df = df.drop('service',axis=1)
df = df.drop('count_ftp_cmd',axis=1)
df = df.drop('attack_cat',axis=1)

print("most data processing done")
#Scaling data

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def preprocess(data,scaling=None):
    data = data.astype(np.float)
    if(scaling == None):
        scaling = StandardScaler()
        datat=scaling.fit_transform(data)
    else:
        datat=scaling.transform(data)
    return(datat,scaling)

Y = df['attack_cat_int']
X = df.drop('attack_cat_int',axis=1)

X_scaled, scaling = preprocess(X.values)
print(X.shape)
print(X_scaled.shape)
print(Y.shape)

print("scaling complete")
#Splitting data

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.1, random_state = 10)

model = tf.keras.models.load_model('NoHiddenModelHPC')
predictions = model.predict(X_test)
Y_pred = predictions.argmax(axis=1)
Y_test_c = Y_test.values.tolist()

pickle.dump(Y_pred, open('../Y_predictions.pkl','wb'))
pickle.dump(Y_test_c, open('../Y_predictions.pkl','wb'))
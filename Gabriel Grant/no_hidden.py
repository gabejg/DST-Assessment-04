import keras
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense
import gzip
import pickle

fp=gzip.open('X_train.pkl.gz','rb')
X_train=pickle.load(fp)
fp.close()
fp=gzip.open('X_test.pkl.gz','rb')
X_test=pickle.load(fp)
fp.close()
fp=gzip.open('Y_train.pkl.gz','rb')
Y_train=pickle.load(fp)
fp.close()
fp=gzip.open('Y_test.pkl.gz','rb')
Y_test=pickle.load(fp)
fp.close()

def compile_fit(model, max_epochs, step_size):
    
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    epochs_list=list()
    trainacc=list()
    testacc=list()
    
    epochs = 0
    
    for i in range(0, int(max_epochs / step_size)):
        
        print("epoch : " + str(epochs))
        
        model.fit(X_train, Y_train, epochs=step_size, batch_size=256, validation_data=(X_test, Y_test))
        trainscores = model.evaluate(X_train, Y_train)
        testscores = model.evaluate(X_test, Y_test)
        
        trainacc.append(trainscores[1])
        testacc.append(testscores[1])
        epochs = epochs + step_size
        epochs_list.append(epochs)
        
    return epochs_list, trainacc, testacc

# Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

model = Sequential()
model.add(Dense(24, input_dim=X_train.shape[1], activation='relu'))   
model.add(Dense(10, activation='softmax'))


epochs, trainacc, testacc = compile_fit(model, 1000, 2) # HPC version


model.save('NoHiddenModelHPC')
pickle.dump(epochs, open('epochs_nohidden.pkl','wb'))
pickle.dump(trainacc, open('trainacc_nohidden.pkl','wb'))
pickle.dump(testacc, open('testacc_nohidden.pkl','wb'))
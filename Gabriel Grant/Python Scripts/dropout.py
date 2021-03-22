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

def create_model(hidden_layers, dropout_perc, init_mode='uniform'):

    model = Sequential()
    model.add(Dense(24, input_dim=X_train.shape[1], activation='relu'))
    
    for i in range(int(hidden_layers)):
        model.add(Dropout(dropout_perc, activation='relu'))
    
    model.add(Dense(10, kernel_initializer=init_mode, activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


modelHT = KerasClassifier(build_fn=create_model, verbose=0)

hidden_layers = [1,2,3,4,5,6,7,8,9,10]
dropout_perc = [.1,.2,.3,.4,.5]
param_grid = dict(hidden_layers=hidden_layers, dropout_perc = dropout_perc, batch_size = [256], epochs=[10])

grid = GridSearchCV(estimator=modelHT, param_grid=param_grid, scoring='accuracy')
grid_result = grid.fit(X_train, Y_train)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']].to_csv('Grid_Optimisation.csv')


# Recreating with our best parameters

hl = grid.best_params_['hidden_layers']
best_drop = grid.best_params_['dropout_perc']

tuned_model = create_model(hidden_layers = hl,dropout_perc = best_drop)

epochs, trainacc, testacc = compile_fit(tuned_model, 1000, 2) # HPC version

tuned_model.save('DropoutModelHPC')
pickle.dump(epochs, open('epochs_dropout.pkl','wb'))
pickle.dump(trainacc, open('trainacc_dropout.pkl','wb'))
pickle.dump(testacc, open('testacc_dropout.pkl','wb'))
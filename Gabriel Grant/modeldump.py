import pickle
import tensorflow as tf
import gzip

fp=gzip.open('X_test.pkl.gz','rb')
X_test=pickle.load(fp)
fp.close()

X_test = pickle.load(open("X_test.pkl.gz

model = tf.keras.models.load_model('NoHiddenModelHPC')
predictions = model.predict(X_test)
Y_pred = predictions.argmax(axis=1)
Y_test_c = Y_test.values.tolist()

pickle.dump(Y_pred, open('../Y_predictions.pkl','wb'))
pickle.dump(Y_test_c, open('../Y_predictions.pkl','wb'))
import numpy as np
import pandas as pd 

train = pd.read_csv('criminal_train.csv')
test = pd.read_csv('criminal_test.csv')

person_id = test['PERID']
del train['PERID']
del test['PERID']

y = train['Criminal'].values
del train['Criminal']


X_train = train.values
X_test = test.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_dim = 55
from sklearn.decomposition import PCA
pca = PCA(n_components = input_dim)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras import optimizers

model = Sequential()

model.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu', input_dim = input_dim))
model.add(Dropout(0.2))
model.add(Dense(init = 'uniform', activation = 'relu', output_dim = 64))
model.add(Dropout(0.4))
model.add(Dense(init = 'uniform', activation = 'relu', output_dim = 128))
model.add(Dropout(0.5))
model.add(Dense(init = 'uniform', activation = 'relu', output_dim = 256))
model.add(Dropout(0.5))
model.add(Dense(init = 'uniform', activation = 'relu', output_dim = 128))
model.add(Dropout(0.4))
model.add(Dense(init = 'uniform', activation = 'relu', output_dim = 64))
model.add(Dense(init = 'uniform', activation = 'sigmoid', output_dim = 2))

sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

model.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()
y = to_categorical(y)
model.fit(X_train,y, batch_size = 16, epochs = 100)

y_pred = model.predict(X_test)

ans = []
y_pred = np.array(y_pred)
for i,j in y_pred:
	if i < j:
		ans.append(1)
	else:
		ans.append(0)

y_pred = np.array(ans)

def get_csv(y_pred):
    import csv
    ans = np.append(pd.DataFrame(person_id) ,pd.DataFrame(y_pred), axis = 1)
    filename = 'CriminalNN.csv'
    header = ['PERID','Criminal']
    ans = list(ans)
    with open(filename,'w') as csvfile:
    	csv_writer = csv.writer(csvfile)
    	csv_writer.writerow(header)
    	csv_writer.writerows(ans)

get_csv(y_pred)

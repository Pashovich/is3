import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(path):
    xdata = []
    ydata = []
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path,file),'r') as f:
            temp = f.read().replace('\n', ' ')
            xdata.append([int(i) for i in temp.split()])
            ydata.append(file)
    return np.array(xdata), ydata

def multiply(x,y, num=1000):
    final_x = []
    final_y = []
    for temp_x,temp_y in zip(x,y):
        for i in range(num):
            final_x.append(temp_x)
            final_y.append(temp_y)
    return np.array(final_x),np.array(final_y)


x_data ,y_data = load_data('data')
x,y = multiply(x_data,y_data)

y = np.array(y).reshape((-1,1))

encoder = OneHotEncoder(categories='auto')
labels_np_onehot = encoder.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(x, labels_np_onehot)

print(X_train.shape)
print(y_train.shape)

print(y_train.shape[1])
model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(35,), units=32, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.summary()

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=100, batch_size=128)

print(model.evaluate(X_test, y_test))
def print_dat(temp):
    print(np.array(temp).reshape(7,5))
for dat in x_data:
    print_dat(dat)
    b = []
    b.append(dat)
    c = []
    c.append(b)
    predicted_outputs = model.predict(c)
    print(predicted_outputs)
    print(np.argmax(predicted_outputs, axis=1))
    # expected_outputs = np.argmax(y_test, axis=1)

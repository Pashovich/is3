import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import argparse


def load_file(file, path = './'):
    with open(os.path.join(path,file),'r') as f:
        temp = f.read().replace('\n', ' ')
    return temp

def load_data(path):
    xdata = []
    ydata = []
    files = os.listdir(path)
    for file_name in files:
            temp = load_file(file_name,path)
            xdata.append([int(i) for i in temp.split()])
            ydata.append(file_name)
    return np.array(xdata), ydata

def multiply(x,y, num=1000):
    final_x = []
    final_y = []
    for temp_x,temp_y in zip(x,y):
        for i in range(num):
            final_x.append(temp_x)
            final_y.append(temp_y)
    return np.array(final_x),np.array(final_y)



def print_matrix(temp):
    print(np.array(temp).reshape(7,5))

def train():
    x_data ,y_data = load_data('data')
    x,y = multiply(x_data,y_data)

    y = np.array(y).reshape((-1,1))

    encoder = OneHotEncoder(categories='auto')
    labels_np_onehot = encoder.fit_transform(y).toarray()

    X_train, X_test, y_train, y_test = train_test_split(x, labels_np_onehot)
    print(X_train.shape)
    model = keras.Sequential()
    model.add(keras.layers.Dense(32,input_dim = 35,  activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data = (X_test, y_test))
    for dat in x_data:
        print_matrix(dat)
        predicted_outputs = model.predict(np.asarray([dat]))
        print(np.argmax(predicted_outputs, axis=1))
    model.save('model.h5')

def test():
    model = keras.models.load_model('model.h5')
    data = load_file('testfile')
    number =  [int(i) for i in data.split()]
    predicted_outputs = model.predict(np.asarray([number]))
    print_matrix(number)
    print(np.argmax(predicted_outputs, axis=1))
    

funcs = {
    "train" : train,
    "test" : test
}

def main():
    parser = argparse.ArgumentParser(description='Modes')
    parser.add_argument('-m', action="store", dest="mode")
    args = parser.parse_args()
    funcs[args.mode]()


if __name__ == "__main__":
    main()



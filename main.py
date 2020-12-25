import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import argparse


def load_file(file, path = './'):
    #загрузка одного файла
    with open(os.path.join(path,file),'r') as f:
        temp = f.read().replace('\n', ' ')
    return temp

def load_data(path):
    #загрузка всех файлов
    xdata = []
    ydata = []
    files = os.listdir(path)
    for file_name in files:
            temp = load_file(file_name,path)
            xdata.append([int(i) for i in temp.split()])
            ydata.append(file_name)
    return xdata, ydata

def multiply(x,y, num=1000):
    #размножение датасета
    final_x = []
    final_y = []
    for temp_x,temp_y in zip(x,y):
        for i in range(num):
            final_x.append(temp_x)
            final_y.append(temp_y)
    return np.array(final_x),np.array(final_y)

def predict(model,number):
    #метод предсказаний
    return model.predict(np.asarray([number]))

def print_matrix(temp):
    #вывод матрицы
    print(np.array(temp).reshape(7,5))

def train():
  #Функция тренировки модели
    x_data ,y_data = load_from_numbers()# Загружаем данные
    x,y = multiply(x_data,y_data)# Увеличиваем размер датасета

    y = np.array(y).reshape((-1,1))

    encoder = OneHotEncoder(categories='auto')
    labels_transformed = encoder.fit_transform(y).toarray()#Кодируем данные в OneHot вектор


    X_train, X_test, y_train, y_test = train_test_split(x, labels_transformed)#Разделяем данные на тренировочный и обучающий наборы
    #Описываем модель
    input_layer = keras.layers.Input(shape=(35,))
    d_1 = keras.layers.Dense(32, activation='relu')(input_layer)
    d_2 = keras.layers.Dense(16, activation='relu')(d_1)
    d_2 = keras.layers.Dense(y_train.shape[1], activation='softmax')(d_2)

    model = keras.models.Model(input_layer,d_2)
    model.summary()
    #Выбираем оптимизатор и метрики
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    #Подаем данные для обучения
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data = (X_test, y_test))
    for dat in x_data:
        print_matrix(dat)
        predicted_outputs = predict(model,dat)
        print('predicted number is',np.argmax(predicted_outputs, axis=1))
    model.save('model.h5')

def test():
    #функция тестирования
    model = keras.models.load_model('model.h5') #загрузка модели
    data = load_file('testfile')
    number =  [int(i) for i in data.split()]
    predicted_outputs = predict(model,number)
    print_matrix(number)
    print('predicted number is',np.argmax(predicted_outputs, axis=1))
    

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



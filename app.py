#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Загружаем необходимые библиотеки
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# In[4]:


#os.system('cls||clear') - очистка окна консоли
os.system('cls||clear')
print('Введите путь до файла с данными для обучения модели:') 


# In[8]:


#Вводим путь до файла с данными вместе с названием и расширением (датасет)
file_parh = input()


# In[26]:


#Загружаем датасет
data = pd.read_excel(file_parh)


# In[ ]:


#Информационное сообщение о прогрессе
os.system('cls||clear')
print('Идет предобработка данных (исключение выбросов)')


# In[27]:


#После загрузки данных проводим исключение выбросов в автоматическом режиме (трижды)
#По всем столбцам, для которых есть выбросы, сделаем замену выбросов на пустые значения
k = 0
while k < 3:
    i = 0
    while i < len(data.columns):
        x = data.columns[i]
        q75,q25 = np.percentile(data.loc[:,x],[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        data.loc[data[x]<min,x] = np.nan
        data.loc[data[x]>max,x] = np.nan
        i += 1
    #Исключим те строки, которые содержат выбросы (пустые значения по некоторым столбцам)
    data = data.dropna(axis=0)
    k += 1


# In[30]:


#Информационное сообщение о прогрессе
os.system('cls||clear')
print('Идет нормализация данных')


# In[31]:


#Нормализуем данные (приведем к диапазону [0,1])
from sklearn import preprocessing
minmaxscalar = preprocessing.MinMaxScaler()
col = data.columns
result = minmaxscalar.fit_transform(data)
minmaxresult = pd.DataFrame(result, columns=col)


# In[ ]:


#Информационное сообщение о прогрессе
os.system('cls||clear')
print('Идет настройка библиотек')


# In[ ]:


#Импортируем библиотеки для построения моделей
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#Импорт TensorFlow
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# In[44]:


#Список моделей для пронозирования
models = ["LinearRegression", "KNeighborsRegressor", "GradientBoostingRegressor", "NeuralNetwork"]


# In[45]:


#Объявляем функцию для обучения модели линейной регрессии. На входе в функцию обучающая выборка, на выходе модель
def LRmodel(test_features, test_labels):
    #Подставляем оптимальные гиперпараметры в модель
    model_base = LinearRegression(positive=True)
    #Обучаем модель
    model_base.fit(test_features,test_labels)
    return model_base


# In[50]:


#Объявляем функцию для обучения модели ближайших соседей. На входе в функцию обучающая выборка, на выходе модель
def KNRmodel(test_features, test_labels):
    #Подставляем оптимальные гиперпараметры в модель
    model_base = KNeighborsRegressor(algorithm='brute', leaf_size=10, n_neighbors=100, weights='distance')
    #Обучаем модель
    model_base.fit(test_features,test_labels)
    return model_base


# In[46]:


#Объявляем функцию для обучения модели градиентного бустинга. На входе в функцию обучающая выборка, на выходе модель
def GBRmodel(test_features, test_labels):
    #Подставляем оптимальные гиперпараметры в модель
    model_base = GradientBoostingRegressor(loss='lad', max_depth=2)
    #Обучаем модель
    model_base.fit(test_features,np.ravel(test_labels))
    return model_base


# In[47]:


#Объявляем функцию для обучения нейросети. На входе в функцию обучающая выборка, тестовая выборка, список входных параметров
#список выходных параметров, датасет, на выходе сама нейросеть. Но в самой функции происходит также проверка сети на тестовых данных
#Ввод параметров для оценки целевой переменной и вывод прогнозного значения
def NNmodel(test_features, test_labels, Xtest1, Ytest1, colni, colno, data_v):
    #Переформатируем данные в массив
    trgn_data = test_labels.values
    trnn_data = test_features.values
    trgn_data = np.ravel(trgn_data)
    Xtrnn = trnn_data 
    Ytrnn = trgn_data
    #Объявляем плейсхолдеры
    X = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    #Инициализаторы
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()
    #Параметры архитектуры модели
    n_start = 10
    n_neurons_1 = 32
    n_neurons_2 = 16
    n_neurons_3 = 8
    n_neurons_4 = 4
    n_target = 1
    #Уровень 1: Переменные для скрытых весов и смещений
    W_hidden_1 = tf.Variable(weight_initializer([n_start, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    #Уровень 2: Переменные для скрытых весов и смещений
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    #Уровень 3: Переменные для скрытых весов и смещений
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    #Уровень 4: Переменные для скрытых весов и смещений
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    #Уровень выходных данных: Переменные для скрытых весов и смещений
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))
    #Скрытый уровень
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    #Выходной уровень (должен быть транспонирован)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    #Функция стоимости
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    #Оптимизатор
    opt = tf.train.AdamOptimizer().minimize(mse)
    #Создание сессии
    netn2 = tf.Session()
    #Запуск инициализатора
    netn2.run(tf.global_variables_initializer())

    #Количество эпох и размер куска данных
    epochs = 100
    batch_size = 50

    for e in range(epochs):

        #Перемешивание данных для обучения
        shuffle_indices = np.random.permutation(np.arange(len(Ytrnn)))
        Xtrnn = Xtrnn[shuffle_indices]
        Ytrnn = Ytrnn[shuffle_indices]

        #Обучение мини-партией
        for i in range(0, len(Ytrnn) // batch_size):
            start = i * batch_size
            batch_x = Xtrnn[start:start + batch_size]
            batch_y = Ytrnn[start:start + batch_size]
            netn2.run(opt, feed_dict={X: batch_x, Y: batch_y})
    #Оцениваем точность на тестовом наборе
    pred1 = netn2.run(out, feed_dict={X: Xtest1})
    predict = np.reshape(pred1,(pred1.size, 1))
    errors = abs(predict - Ytest1)
    print('Средняя абсолютная ошибка оценки параметра', end =" ")
    print(np.mean(errors))
    #Вводим и нормализуем данные, по которым будет осуществляться прогноз
    imp_values = InputValues(colni, data_v)
    #Осуществляем прогноз
    predict_label = netn2.run(out, feed_dict={X: imp_values})
    #Преобразуем из нормализованных данных в стандартные
    col = colno
    #Определим параметры, которые использовались для нормализации
    minv = np.min(data_v[col])
    maxv = np.max(data_v[col])
    predict_label[0] = predict_label[0]*(maxv - minv) + minv
    print(f"Прогнозное значение параметра {col[0]} составляет {predict_label[0][0]}")
    return netn2


# In[48]:


#Определяем функцию для вычисления точности модели. На входе модель, а также входные параметры и целевая переменная
def evaluate(model, test_features, test_labels):
    #Делаем предсказание на основе входных параметров
    predictions = model.predict(test_features)
    #Считаем абсолютные ошибки в предсказаниях (разность между предсказанным значением и целевым значением)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    #Определяем точность модели
    accuracy = 100 - mape
    print('Средняя абсолютная ошибка: {:0.4f}'.format(np.mean(errors.values)))
    #print('Точность = {:0.2f}%.'.format(accuracy[0]))
    return accuracy


# In[49]:


#Определяем функцию для вычисления точности модели. На входе модель, а также входные параметры и целевая переменная
def evaluate_2(model, test_features, test_labels):
    #Делаем предсказание на основе входных параметров
    predictions = model.predict(test_features)
    #Преобразуем к виду [[],[],...] из одномерного массива
    predict = np.reshape(predictions,(predictions.size, 1))
    #Считаем абсолютные ошибки в предсказаниях (разность между предсказанным значением и целевым значением)
    errors = abs(predict - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    #Определяем точность модели
    accuracy = 100 - mape
    print('Средняя абсолютная ошибка: {:0.4f}'.format(np.mean(errors.values)))
    #print('Точность = {:0.2f}%.'.format(accuracy[0]))
    return accuracy


# In[55]:


#Определяем функцию, которая будет использоваться для заведения входных параметров. На входе список параметров,  датасет
#На выходе получаем датафрейм с введенными параметрами
def InputValues(cols, maindata):
    #Создаем датафрем для входных значений
    input_val = pd.DataFrame()
    i = 0
    while i < len(cols):
        a = []
        line = f"Введите значение параметра ({cols[i]}): "
        param_val = input(line)
        param_value = float(param_val)
        #Проводим нормализацию данных на начальных (входных) данных модели
        minv = np.min(maindata[cols[i]])
        maxv = np.max(maindata[cols[i]])
        param_value = (param_value - minv)/(maxv - minv)
        a.append(param_value)
        input_val[cols[i]] = a
        i += 1
    return input_val


# In[ ]:


exitway = 'Y' #Переменная для выхода из приложения
while exitway != 'N':
    os.system('cls||clear')
    print('Выберите целевую переменную для прогнозирования из предложенного списка:')
    colnames_in = []
    #выводим список доступных параметров
    i = 0
    while i < len(data.columns):
        print(data.columns[i])
        colnames_in.append(data.columns[i])
        i += 1
    label = input() #вводим значение
    colnames_out = []
    colnames_out.append(label)
    colnames_in.remove(label) #исключаем выбранный параметр из общего списка
    param = 'Y' #переменная для выхода для цикла ниже
    #цикл для исключения параметров из списка входных
    while param != 'N':
        os.system('cls||clear')
        print('Выберите переменную для исключения из списка параметров, либо введите N для завершения исключения параметров:')
        #выводим список доступных параметров
        k = 0
        while k < len(colnames_in):
            print(colnames_in[k])
            k += 1
        param = input()
        if param != 'N':
            colnames_in.remove(param)  #исключаем выбранный параметр из общего списка  
    k = 0
    #Разделим параметры
    #Выходные
    trg = minmaxresult[colnames_out]
    #Входные
    trn = minmaxresult[colnames_in]
    #Подготовка обучающей и тестовой выборок (соотношение 70 на 30)
    Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.3)
    os.system('cls||clear')
    print('Для продолжения нажмите любую клавишу, для переподбора параметров нажмите Y')
    exitway = input()
    if exitway != 'Y':
        os.system('cls||clear')
        print('Выберите прогнозную модель из списка:')
        #выводим список прогнозных моделей
        k = 0
        while k < len(models):
            print(models[k])
            k += 1
        model_name = input()
        #информационное сообщение
        print('Идет обучение модели')
        if model_name == 'LinearRegression':
            model = LRmodel(Xtrn, Ytrn)
            os.system('cls||clear')
            #Оцениваем точность на тестовом наборе
            base_accuracy = evaluate(model, Xtest, Ytest)
            #Цикл для многократного введения параметров и оценки выходной переменной
            endparam = 'Y'
            while endparam != 'END':
                #Вводим и нормализуем данные, по которым будет осуществляться прогноз
                imp_values = InputValues(colnames_in, data)
                #Осуществляем прогноз
                predict_label = model.predict(imp_values)
                #Преобразуем из нормализованных данных в стандартные
                col = colnames_out
                #Определим параметры, которые использовались для нормализации
                minv = np.min(data[col])
                maxv = np.max(data[col])
                #преобразуем к изначальному виду
                predict_label[0] = predict_label[0]*(maxv - minv) + minv
                print(f"Прогнозное значение параметра {col[0]} составляет {predict_label[0][0]}")
                print("Для продолжения нажмите любую кнопку, для завершения введите END")
                endparam = input()
        elif model_name == 'KNeighborsRegressor':
            model = KNRmodel(Xtrn, Ytrn)
            os.system('cls||clear')
            #Оцениваем точность на тестовом наборе
            base_accuracy = evaluate(model, Xtest, Ytest)
            #Цикл для многократного введения параметров и оценки выходной переменной
            endparam = 'Y'
            while endparam != 'END':
                #Вводим и нормализуем данные, по которым будет осуществляться прогноз
                imp_values = InputValues(colnames_in, data)
                #Осуществляем прогноз
                predict_label = model.predict(imp_values)
                #Преобразуем из нормализованных данных в стандартные
                col = colnames_out
                #Определим параметры, которые использовались для нормализации
                minv = np.min(data[col])
                maxv = np.max(data[col])
                #преобразуем к изначальному виду
                predict_label[0] = predict_label[0]*(maxv - minv) + minv
                print(f"Прогнозное значение параметра {col[0]} составляет {predict_label[0][0]}")
                print("Для продолжения нажмите любую кнопку, для завершения введите END")
                endparam = input()
        elif model_name == 'GradientBoostingRegressor':
            model = GBRmodel(Xtrn, Ytrn)
            os.system('cls||clear')
            #Оцениваем точность на тестовом наборе
            base_accuracy = evaluate_2(model, Xtest, Ytest)
            #Цикл для многократного введения параметров и оценки выходной переменной
            endparam = 'Y'
            while endparam != 'END':
                #Вводим и нормализуем данные, по которым будет осуществляться прогноз
                imp_values = InputValues(colnames_in, data)
                #Осуществляем прогноз
                predict_label = model.predict(imp_values)
                #Преобразуем из нормализованных данных в стандартные
                col = colnames_out
                #Определим параметры, которые использовались для нормализации
                minv = np.min(data[col])
                maxv = np.max(data[col])
                #преобразуем к изначальному виду
                predict_label[0] = predict_label[0]*(maxv - minv) + minv
                print(f"Прогнозное значение параметра {col[0]} составляет {predict_label[0]}")
                print("Для продолжения нажмите любую кнопку, для завершения введите END")
                endparam = input()
        elif model_name == 'NeuralNetwork':
            os.system('cls||clear')
            model = NNmodel(Xtrn, Ytrn, Xtest, Ytest, colnames_in, colnames_out, data)
        else:
            os.system('cls||clear')
            print('Ошибка при выборе модели, вернитесь к подбору параметров')
        print('Для возвращения к выбору параметров нажмите любую клавишу. Для завершения введите N')
        exitway = input()


# In[ ]:




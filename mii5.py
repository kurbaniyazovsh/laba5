import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



dataset = pd.read_csv("/Users/katyaanosova/Desktop/Нужно разобрать/МИИ/МИИ-5/Курбаниязов/diabetes.csv")
#Подготовка данных к обучению
target = dataset["Outcome"]
dataset.drop("Outcome", axis = 1, inplace = True)

#Разделение данных на выборки
X_train, X_test, Y_train, Y_test = train_test_split(dataset, target, random_state=0)

#Обучение на сырых данных
#Случайный лес
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, Y_train)
predictions = rf_model.predict(X_test)
print("Точность классификатора: {}%".format((rf_model.score(X_test,Y_test))*100))
#Стохастический градиентный спуск
sgd = SGDClassifier (loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
predictions = sgd.predict(X_test)
print("Точность классификатора: {}%".format((sgd.score(X_test,Y_test))*100))
#Линейный дискриминантный анализ
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, Y_train)
predictions = lda_model.predict(X_test)
print("Точность классификатора: {}%".format((lda_model.score(X_test,Y_test))*100))

#Обработка данных
scaler = StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)
scaler = Normalizer().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

#Обучение на очищенных данных
#Случайный лес
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, Y_train)
predictions = rf_model.predict(X_test)
print("Точность классификатора: {}%".format((rf_model.score(X_test,Y_test))*100))
#Стохастический градиентный спуск
sgd = SGDClassifier (loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
predictions = sgd.predict(X_test)
print("Точность классификатора: {}%".format((sgd.score(X_test,Y_test))*100))
#Линейный дискриминантный анализ
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, Y_train)
predictions = lda_model.predict(X_test)
print("Точность классификатора: {}%".format((lda_model.score(X_test,Y_test))*100))

#Визуализация
plt.subplot(1, 2, 1)
plt.hist(Y_test)
plt.subplot(1, 2, 2)
plt.hist(predictions)
plt.show()

print(Y_test)
print(predictions)

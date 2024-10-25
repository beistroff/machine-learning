import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Завантаження набору даних Iris
iris = datasets.load_iris()

# Огляд структури даних 
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
print(iris_df.info())
print(iris_df.head())

# Розділення набору даних на тренувальну (70%) та тестову (30%) вибірки
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=123)

# Створення та тренування моделі наївного баєсівського класифікатора
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = nb_classifier.predict(X_test)

# Порівняння прогнозованих результатів з фактичними
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = round(accuracy_score(y_test, y_pred), 4)
report = classification_report(y_test, y_pred)

# Виведення результатів
print("Матриця помилок:\n", conf_matrix)
print("Точність моделі:", accuracy)
print("Звіт по класифікації:\n", report)

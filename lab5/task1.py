import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# Завантажуємо навчальні та тестові дані
train_data = pd.read_csv('data/svmdata1.txt', sep='\t')
test_data = pd.read_csv('data/svmdata1test.txt', sep='\t')

# Розділяємо дані на ознаки (X) та мітки (y)
X_train = train_data[['X1', 'X2']].values
y_train = np.where(train_data['Color'] == 'red', 0, 1)  # Червоний = 0, Зелений = 1

X_test = test_data[['X1', 'X2']].values
y_test = np.where(test_data['Color'] == 'red', 0, 1)

# Створюємо SVM-модель з лінійним ядром
model = svm.SVC(C=1, kernel='linear')
model.fit(X_train, y_train)

# Прогнозуємо результати на навчальній та тестовій вибірці
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Розраховуємо точність і кількість помилок
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_errors = len(y_train) - sum(y_train == y_train_pred)
test_errors = len(y_test) - sum(y_test == y_test_pred)

print(f'Кількість опорних векторів: {len(model.n_support_)}')
print(f'Точність на навчальній вибірці: {train_accuracy * 100:.2f}%')
print(f'Точність на тестовій вибірці: {test_accuracy * 100:.2f}%')
print(f'Кількість помилок на навчальній вибірці: {train_errors}')
print(f'Кількість помилок на тестовій вибірці: {test_errors}')

# Функція для побудови графіку розбиття простору ознак
def plot_svm_decision_boundary(model, X, y):
    h = 0.02  # Крок сітки
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Відображаємо область рішення
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Границя рішення SVM (лінійне ядро)')
    plt.show()

# Візуалізуємо розбиття простору ознак для навчальних даних
plot_svm_decision_boundary(model, X_train, y_train)

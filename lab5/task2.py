import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def task2():
    train_data = pd.read_csv('data/svmdata2.txt', sep='\t')
    X_train = train_data[['X1', 'X2']].values
    y_train = train_data['Colors'].values

    test_data = pd.read_csv('data/svmdata2test.txt', sep='\t')
    X_test = test_data[['X1', 'X2']].values
    y_test = test_data['Colors'].values

    # Кодування кольорів як міток
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Налаштування параметру C для лінійного ядра
    param_grid = {'C': np.logspace(-3, 4, 20)}
    svm_model = SVC(kernel='linear')

    # Використання GridSearchCV для пошуку оптимального C
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Виведення оптимального параметру C
    optimal_C = grid_search.best_params_['C']
    print(f"Оптимальне значення параметра C: {optimal_C}")

    # Побудова моделі з оптимальним C
    best_svm_model = SVC(kernel='linear', C=optimal_C)
    best_svm_model.fit(X_train, y_train)

    # Оцінка точності на навчальній та тестовій вибірках
    train_predictions = best_svm_model.predict(X_train)
    test_predictions = best_svm_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Точність на навчальній вибірці: {train_accuracy * 100:.2f}%")
    print(f"Точність на тестовій вибірці: {test_accuracy * 100:.2f}%")

    def plot_decision_boundary(model, X, y):
        h = 0.01  # Крок сітки
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f"Оптимальне C = {optimal_C}")
        plt.show()

    plot_decision_boundary(best_svm_model, X_test, y_test)

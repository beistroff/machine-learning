import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def task4():
    data = pd.read_csv('data/svmdata4.txt', delimiter='\t')

    # Підготовка даних
    X = data[['X1', 'X2']].values
    y = LabelEncoder().fit_transform(data['Colors'])  # Перетворення кольорів у числові значення

    # Масив для зберігання помилок
    eps_values = np.linspace(0.1, 1.0, 10)
    mse_values = []

    # Побудова моделі для різних значень epsilon
    for eps in eps_values:
        model = SVR(kernel='rbf', C=1, epsilon=eps)
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        mse_values.append(mse)

    # Візуалізація залежності MSE від epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(eps_values, mse_values, marker='o', color='b')
    plt.title('Залежність MSE від значення epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Середньоквадратична помилка (MSE)')
    plt.grid(True)
    plt.show()

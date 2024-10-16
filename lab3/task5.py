import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

longley_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/longley.csv"

def task5():
    longley = pd.read_csv(longley_url)
    # видалення колонки Population
    longley = longley.drop(columns=['Population'])

    n = longley.shape[0]
    train_indices = np.random.choice(n, n // 2, replace=False)  # 50/50 %
    train_data = longley.iloc[train_indices]
    test_data = longley.drop(train_indices)

    # Готуємо незалежну (X) і залежну (y) матриці
    train_x = train_data.drop(columns=['Unemployed']).values  # незалежні змінні
    train_y = train_data['Unemployed'].values  # залежні змінні

    test_x = test_data.drop(columns=['Unemployed']).values
    test_y = test_data['Unemployed'].values

    # Генерація значень λ для гребневої регресії.
    lambda_values = 10 ** (-3 + 0.2 * np.arange(26))

    train_errors = []
    test_errors = []

    # Будуємо модель гребневої регресії та обчислюємо помилки для кожного значення лямбда
    for lam in lambda_values:
        ridge_model = Ridge(alpha=lam)
        ridge_model.fit(train_x, train_y)  # Підбираємо модель
        train_preds = ridge_model.predict(train_x)
        train_errors.append(mean_squared_error(train_y, train_preds)) # Вираховуємо помилки під час тренування 

        test_preds = ridge_model.predict(test_x)
        test_errors.append(mean_squared_error(test_y, test_preds))

    # Перетворюємо на масиви numpy для легшого індексування
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    # Будуємл MSE на навчальних і тестових наборах
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, train_errors, label='Train Error', color='yellow', marker='o')
    plt.plot(lambda_values, test_errors, label='Test Error', color='blue', marker='o')
    plt.xscale('linear')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.title('Error on Training and Test Sets')
    plt.legend()
    plt.grid()
    plt.show()

    min_lambda_index = np.argmin(test_errors)

    # Виводимо відповідне значення лямбда та мінімальну помилку
    best_lambda = lambda_values[min_lambda_index]
    print("Значення λ з найменшою помилкою:", best_lambda)
    print("Найменша помилка на тестовій вибірці:", test_errors[min_lambda_index])

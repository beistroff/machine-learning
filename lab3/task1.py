import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import colors

data = pd.read_csv('data/REGLAB1.txt', delimiter='\t')

def task1():
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)

    # Функція для створення моделей
    def build_model(formula, train_data, test_data):
        model = sm.OLS.from_formula(formula, data=train_data).fit()
        model_test = sm.OLS.from_formula(formula, data=test_data).fit()
        return model, model_test

    # Створюємо моделі лінійної регресії
    model_1, model_2 = build_model('y ~ x + z', train_data, test_data) # на основі незалежних x та y
    model_3, model_4 = build_model('y ~ x', train_data, test_data) # залежна змінна y на основі незалежної x
    model_5, model_6 = build_model('y ~ z', train_data, test_data)
    model_7, model_8 = build_model('y ~ x + I(x**2) + I(z**2) + x*z', train_data, test_data)

    # Скориговані коефіцієнтом детермінації
    print(f"{colors.YELLOW}Adjusted R-squared values:{colors.RESET}")
    print('Model 1:', round(model_1.rsquared_adj, 6))
    print('Model 2:', round(model_2.rsquared_adj, 6))
    print('Model 3:', round(model_3.rsquared_adj, 6))
    print('Model 4:', round(model_4.rsquared_adj, 6))
    print('Model 5:', round(model_5.rsquared_adj, 6))
    print('Model 6:', round(model_6.rsquared_adj, 6))
    print('Model 7:', round(model_7.rsquared_adj, 6))
    print('Model 8:', round(model_8.rsquared_adj, 6))

    # Прогнози на тестовій вибірці
    predictions_1 = model_1.predict(test_data)
    predictions_2 = model_2.predict(test_data)
    predictions_3 = model_3.predict(test_data)
    predictions_4 = model_4.predict(test_data)
    predictions_5 = model_5.predict(test_data)
    predictions_6 = model_6.predict(test_data)
    predictions_7 = model_7.predict(test_data)
    predictions_8 = model_8.predict(test_data)

    # Обчислюємо середньо-квадратичну помилку (MSE)
    mse_1 = np.mean((test_data['y'] - predictions_1) ** 2)
    mse_2 = np.mean((test_data['y'] - predictions_2) ** 2)
    mse_3 = np.mean((test_data['y'] - predictions_3) ** 2)
    mse_4 = np.mean((test_data['y'] - predictions_4) ** 2)
    mse_5 = np.mean((test_data['y'] - predictions_5) ** 2)
    mse_6 = np.mean((test_data['y'] - predictions_6) ** 2)
    mse_7 = np.mean((test_data['y'] - predictions_7) ** 2)
    mse_8 = np.mean((test_data['y'] - predictions_8) ** 2)

    print(f"{colors.GREEN}\nMean Squared Errors:{colors.RESET}")
    print("Model 1 MSE:", round(mse_1, 6))
    print("Model 2 MSE:", round(mse_2, 6))
    print("Model 3 MSE:", round(mse_3, 6))
    print("Model 4 MSE:", round(mse_4, 6))
    print("Model 5 MSE:", round(mse_5, 6))
    print("Model 6 MSE:", round(mse_6, 6))
    print("Model 7 MSE:", round(mse_7, 6))
    print("Model 8 MSE:", round(mse_8, 6))

    # Зібрані MSE і Adjusted R-squared у вигляді списків
    mse_list = [mse_1, mse_2, mse_3, mse_4, mse_5, mse_6, mse_7, mse_8]
    rsquared_adj_list = [
        model_1.rsquared_adj, model_2.rsquared_adj,
        model_3.rsquared_adj, model_4.rsquared_adj,
        model_5.rsquared_adj, model_6.rsquared_adj,
        model_7.rsquared_adj, model_8.rsquared_adj
    ]

    # Знаходимо індекси найкращої моделі за MSE і Adjusted R-squared
    best_mse_index = mse_list.index(min(mse_list)) + 1
    best_rsq_index = rsquared_adj_list.index(max(rsquared_adj_list)) + 1

    # Порівнюємо і виводимо найкращу модель
    if best_mse_index == best_rsq_index:
        print(f"Найкраща модель за обома критеріями: Модель {best_mse_index}")
    else:
        print(f"Найкраща модель за MSE: Модель {best_mse_index}")
        print(f"Найкраща модель за Adjusted R-squared: Модель {best_rsq_index}")

    # Заповнюємо пропущені значення середнім по змінній z
    # data['z'].fillna(data['z'].mean(), inplace=True)
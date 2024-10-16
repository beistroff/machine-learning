import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import colors

data = pd.read_csv('data/REGLAB2.txt', delimiter='\t')

def task2():
    # 70% навчальні, 30% тестові
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Функція для розрахунку RSS і MSE
    def calculate_rss_and_mse(model, test_data, dependent_var):
        predictions = model.predict(test_data)
        true_values = test_data[dependent_var]
        squared_differences = (true_values - predictions) ** 2
        rss = np.sum(squared_differences)
        mse = np.mean(squared_differences)
        return rss, mse

    models = {}
    results = []

    # Додавання постійного члена для перехоплення в регресійних моделях
    train_data = sm.add_constant(train_data)
    test_data = sm.add_constant(test_data)

    # Визначення регресійних моделей ~
    formulas = [
        'y ~ x1',
        'y ~ x2',
        'y ~ x3',
        'y ~ x4',
        'y ~ x1 + x2',
        'y ~ x1 + x3',
        'y ~ x1 + x4',
        'y ~ x2 + x3',
        'y ~ x2 + x4',
        'y ~ x3 + x4',
        'y ~ x1 + x2 + x3',
        'y ~ x1 + x2 + x4',
        'y ~ x1 + x3 + x4',
        'y ~ x2 + x3 + x4',
        'y ~ x1 + x2 + x3 + x4'
    ]

    # Цикл для підбору моделей і обчислення RSS і MSE
    for formula in formulas:
        model = sm.OLS.from_formula(formula, data=train_data).fit()
        rss, mse = calculate_rss_and_mse(model, test_data, 'y')
        models[formula] = model
        results.append({'Model': formula, 'RSS': rss, 'MSE': mse})

    results_df = pd.DataFrame(results)

    print(f"{colors.BLUE}Залишкова сума квадратів і середньо-квадратична похибка  для кожної моделі:\n{colors.RESET}")
    print(results_df)

    # Знаходимо найкращу модель на основі найнижчого RSS
    best_model_row = results_df.loc[results_df['RSS'].idxmin()]
    best_model_name = best_model_row['Model']
    best_model = models[best_model_name]

    print(f"{colors.GREEN}\nНайкраща модель: {colors.RESET}{best_model_name}")
    print(f"{colors.YELLOW}\nПідсумковий звіт моделі:\n{colors.RESET}")
    print(best_model.summary())

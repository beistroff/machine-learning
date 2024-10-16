import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/ALIGATORS.txt", delimiter='\t') 

def task4():
    """If you find that 5 bins don't adequately capture the distribution 
    of weights (e.g., if some bins are empty), you may want to increase or 
    decrease the number of bins accordingly."""
    bins = pd.cut(data['Weight'], bins=5)  # Adjust the number of bins based on your data distribution

    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)

    # Групування даних за категоріями
    for _, group in data.groupby(bins):
        train_sample = group.sample(frac=0.7, random_state=42)  # 70% для тренування
        test_sample = group.drop(train_sample.index)
        train_data = pd.concat([train_data, train_sample])
        test_data = pd.concat([test_data, test_sample])

    def train_and_evaluate_model(degree, train_data, test_data):
        poly = PolynomialFeatures(degree=degree)

        # Перетворення незалежної змінної довжини на поліноміальні форми для тренувальних та тестових даних.
        X_train_poly = poly.fit_transform(train_data[['Length']])
        X_test_poly = poly.transform(test_data[['Length']])
        
        model = LinearRegression()
        model.fit(X_train_poly, train_data['Weight'])
        
        predictions = model.predict(X_test_poly)

        # Обчислення залишкової суми квадратів (RSS)
        rss = np.sum((test_data['Weight'] - predictions) ** 2)
        mse = mean_squared_error(test_data['Weight'], predictions)
        
        return model, rss, mse, predictions

    # Встановлення лінійної моделі
    linear_model, rss_linear, mse_linear, predictions_linear = train_and_evaluate_model(1, train_data, test_data)
    models = {'Linear': linear_model}
    results = pd.DataFrame({'Model': ['Linear'], 'RSS': [rss_linear], 'MSE': [mse_linear]})

    for degree in range(2, 6):
        model, rss, mse, predictions = train_and_evaluate_model(degree, train_data, test_data)
        models[f'Poly {degree}'] = model
        results = pd.concat([results, pd.DataFrame({'Model': [f'Poly {degree}'], 'RSS': [rss], 'MSE': [mse]})], ignore_index=True)

    print("\nЗалишкова сума квадратів (RSS) та середньо-квадратична похибка (MSE) для кожної поліноміальної моделі:\n")
    print(results)

    # Найкраща модель за RSS
    best_model_name = results.loc[results['RSS'].idxmin(), 'Model']
    best_model = models[best_model_name]
    print(f"\nНайкраща модель за RSS: {best_model_name}\n")

    # Побудова графіку
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Length', y='Weight', data=data, color='blue', alpha=0.5, label='Train Data')

    x_vals = np.linspace(min(data['Length']), max(data['Length']), 100).reshape(-1, 1)

    x_poly_linear = PolynomialFeatures(degree=1).fit_transform(x_vals)
    y_vals_linear = models['Linear'].predict(x_poly_linear)
    plt.plot(x_vals, y_vals_linear, color='black', label='Linear Model')

    for degree in range(2, 6):
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x_vals)
        y_vals = models[f'Poly {degree}'].predict(x_poly)
        plt.plot(x_vals, y_vals, label=f'Poly {degree}')

    # Прогнози найкращої моделі
    X_test_poly = PolynomialFeatures(degree=int(best_model_name.split()[1])).fit_transform(test_data[['Length']])
    predictions = best_model.predict(X_test_poly)
    plt.scatter(test_data['Length'], test_data['Weight'], color='red', label='Test Data')
    plt.plot(test_data['Length'], predictions, color='green', linestyle='dashed', label='Predictions')

    plt.title(f'Модель для залежності ваги від довжини ({best_model_name})')
    plt.xlabel('Довжина')
    plt.ylabel('Вага')
    plt.legend()
    plt.grid(True)
    plt.show()

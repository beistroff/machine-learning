import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import colors

data = pd.read_csv("data/CYGAGE.txt", delimiter="\t")

def task3():
    X = data[['Depth']]  # Незалежна змінна
    y = data['calAge']  # Залежна змінна
    weights = data['Weight']  # Ваги

    # Крос-валідація
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(len(y))

    # Цикл по блоках крос-валідації. X незалежна змінна
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        weights_train = weights.iloc[train_index]

        # Модель лінійної регресії з вагами
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        predictions[test_index] = model.predict(X_test)

    # Обчислення RSS та MSE
    rss = np.sum((y - predictions)**2)
    mse = mean_squared_error(y, predictions)

    print(f"\n{colors.PINK}Залишкова сума квадратів (RSS):{colors.RESET}", round(rss, 3))
    print(f"{colors.PINK}Середньо-квадратична похибка (MSE):{colors.RESET}", round(mse, 3))

    # Побудова графіку
    plt.scatter(data['Depth'], data['calAge'], s=data['Weight']*100, color='red', alpha=0.5, label='Train Data')
    plt.plot(data['Depth'], predictions, 'g--', label='Predictions')

    plt.title('Лінійна регресія з вагами')
    plt.xlabel('Глибина залягання')
    plt.ylabel('Вік досліджуваних покладів')
    plt.legend()

    plt.show()

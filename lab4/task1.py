import pandas as pd
from scipy.stats import shapiro

# Завантажуємо dataset iris
from sklearn.datasets import load_iris

data = load_iris()
iris = pd.DataFrame(data.data, columns=data.feature_names)

# Встановлюємо рівень значущості
alpha = 0.05

# Виконуємо тест Шапіро для кожного стовпця ознак
for column in iris.columns:
    stat, p_value, 6 = shapiro(iris[column])
    print(f"Характеристика: {column}")
    print(f"p-value: {p_value}")
    print(f"Статистика W: {stat}")
    
    if p_value > alpha:
        print(f"p-значення більше за {alpha}, тому розподіл {column} нормальний.\n")
    else:
        print(f"p-значення менше або дорівнює {alpha}, тому розподіл {column} не нормальний.\n")

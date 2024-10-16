import pandas as pd
import numpy as np
from scipy.stats import kstest
import statsmodels.api as sm

data = pd.read_csv("ALLCOUNTRIES.txt", sep=r'\s+', header=0) 

filtered_data = data.dropna(subset=['population', 'area'])
filtered_data = filtered_data[filtered_data['area'] > 10]

# Додаємо логарифмічні стовпці
filtered_data['area_log'] = np.log10(np.log10(filtered_data['area']))
filtered_data['population_log'] = np.log10(np.log10(filtered_data['population']))

# Підбираємо лінійну модель
X = sm.add_constant(filtered_data['area_log'])
y = filtered_data['population_log']
model = sm.OLS(y, X).fit()

# Визначаємо функцію для передбачення
def predict_population_log(area_log, model):
    beta_0, beta_1 = model.params
    return beta_0 + beta_1 * area_log

predicted_population_log = predict_population_log(filtered_data['area_log'], model)

# Виконуємо тест Колмогорова-Смирнова
ks_stat, p_value = kstest(filtered_data['population_log'], predicted_population_log)

print(f"p-value тесту Колмогорова-Смірнова: {p_value}")
if p_value > 0.05:
    print("Немає підстав відхиляти нульову гіпотезу: розподіли однакові.")
else:
    print("Нульова гіпотеза відхиляється: розподіли різні.")

example_area_log = np.array([filtered_data['area_log'].min(), filtered_data['area_log'].mean(), filtered_data['area_log'].max()])
example_population_log = predict_population_log(example_area_log, model)

# Відображення результатів перевірки функціонування
example_results = pd.DataFrame({'area_log': example_area_log, 'predicted_population_log': example_population_log})
print(example_results)

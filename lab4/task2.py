import numpy as np
from scipy.stats import kstest, norm, chi2

k_values = [10, 15, 20, 25, 30]

for k in k_values:
    # Генеруємо нормально розподілені випадкові величини з математичним сподіванням k і стандартним відхиленням sqrt(k)
    normal_sample = np.random.normal(loc=k, scale=np.sqrt(k), size=200)
    
    # Генеруємо випадкові величини, що мають розподіл хі-квадрат з k ступенями вільності
    chisq_sample = np.random.chisquare(df=k, size=200)
    
    # Використовуємо тест Колмогорова-Смірнова для перевірки гіпотези про однаковість розподілів
    ks_stat, p_value = kstest(normal_sample, chisq_sample)
    
    print(f"k = {k}")
    print(f"p-value тесту Колмогорова-Смірнова: {p_value}")
    print(f"Статистика W: {ks_stat}")
    
    if p_value > 0.05:
        print(f"Приймається гіпотеза про однаковість розподілів для k = {k}\n")
    else:
        print(f"Відхиляємо гіпотезу про однаковість розподілів для k = {k}\n")

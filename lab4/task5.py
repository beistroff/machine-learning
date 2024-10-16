import pandas as pd
from scipy.stats import shapiro, ttest_ind

data = pd.read_csv("READINGSPEED.txt", delim_whitespace=True)

# Витягуємо дані для різних типів навчання
dra_data = data['Score'][data['LearningType'] == 'DRA']
sc_data = data['Score'][data['LearningType'] == 'SC']

# Виконуємо тест Шапіро на нормалізацію
shapiro_dra = shapiro(dra_data)
shapiro_sc = shapiro(sc_data)

print("Shapiro-Wilk test for DRA data:", shapiro_dra)
print("Shapiro-Wilk test for SC data:", shapiro_sc)

# Виконуємо двовибірковий t-тест з різними альтернативними гіпотезами
alpha = 0.05

# 1. Альтернативна гіпотеза: середні значення рівні (нульова гіпотеза: середні рівні)
t_test_equal = ttest_ind(dra_data, sc_data, alternative='two-sided')

# 2. Альтернативна гіпотеза: середня для DRA більше (нульова гіпотеза: середня SC >= середньої DRA)
t_test_greater = ttest_ind(dra_data, sc_data, alternative='greater')

# 3. Альтернативна гіпотеза: середня для SC більше (нульова гіпотеза: середня DRA >= середньої SC)
t_test_less = ttest_ind(dra_data, sc_data, alternative='less')

print(f"Two-sided t-test statistic: {t_test_equal.statistic}, p-value: {t_test_equal.pvalue}")
print(f"One-sided t-test (DRA > SC): {t_test_greater.statistic}, p-value: {t_test_greater.pvalue}")
print(f"One-sided t-test (SC > DRA): {t_test_less.statistic}, p-value: {t_test_less.pvalue}")

# Перевіряємо гіпотези
if t_test_equal.pvalue > alpha:
    print("Two-sided: No evidence to reject the null hypothesis: reading speeds are the same.")
else:
    print("Two-sided: Reject the null hypothesis: reading speeds are different.")

if t_test_greater.pvalue > alpha:
    print("(DRA > SC): No evidence to reject the null hypothesis: DRA reading speed is not higher.")
else:
    print("(DRA > SC): Reject the null hypothesis: DRA reading speed is higher.")

if t_test_less.pvalue > alpha:
    print("(SC > DRA): No evidence to reject the null hypothesis: SC reading speed is not higher.")
else:
    print("(SC > DRA): Reject the null hypothesis: SC reading speed is higher.")

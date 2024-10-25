import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

nsw74psid1 = pd.read_csv('data/nsw74psid1.csv')

# Видаляємо рядки з пропущеними значеннями
nsw74psid1 = nsw74psid1.dropna()

# Функція для мінімум-максимум нормалізації
def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

# Нормалізація всіх числових змінних у наборі даних
nsw74psid1_normalized = nsw74psid1.copy()
for col in nsw74psid1_normalized.select_dtypes(include=[np.number]):
    nsw74psid1_normalized[col] = normalize(nsw74psid1_normalized[col])

# Розділення даних на тренувальний та тестовий набори (80% тренувальні, 20% тестові)
X = nsw74psid1_normalized.drop(columns='re78')  # Ознаки
y = nsw74psid1_normalized['re78']  # Цільова змінна
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Побудова регресійного дерева
tree_model = DecisionTreeRegressor(random_state=123, max_depth=2)
tree_model.fit(X_train, y_train)

# Візуалізація початкового дерева
plt.figure(figsize=(10, 8))
plot_tree(tree_model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Регресійне дерево для re78 ~ .")
plt.show()

# Прогнозування за допомогою дерева
tree_predictions = tree_model.predict(X_test)

# Обчислення RMSE для дерева
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_predictions))

# Побудова SVM моделі
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Прогнозування за допомогою SVM
svm_predictions = svm_model.predict(X_test)

# Обчислення RMSE для SVM
svm_rmse = np.sqrt(mean_squared_error(y_test, svm_predictions))

# Виведення результатів
print(f"RMSE для SVM моделі: {svm_rmse:.4f}")
print(f"RMSE для регресійного дерева: {tree_rmse:.4f}")

# Візуалізація результатів SVM
results = pd.DataFrame({'Real': y_test, 'Predicted': svm_predictions})
sns.scatterplot(data=results, x='Real', y='Predicted', alpha=0.5)
plt.plot([results['Real'].min(), results['Real'].max()], [results['Real'].min(), results['Real'].max()], 'b--')
plt.title("Прогнозування SVM: Реальні vs Прогнозовані значення")
plt.xlabel("Реальні значення")
plt.ylabel("Прогнозовані значення")
plt.show()

# Перевірка, яка модель краща
if tree_rmse < svm_rmse:
    print("Регресійне дерево має кращу якість моделі.")
else:
    print("SVM має кращу якість моделі.")

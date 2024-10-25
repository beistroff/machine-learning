import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Завантажуємо дані
data = pd.read_csv('data/spam7.csv')

# Поділ даних на тренувальний та тестовий набори (70% тренувальні, 30% тестові)
X = data.drop(columns='yesno')  # Ознаки
y = data['yesno']  # Цільова змінна

# Поділ на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Побудова дерева рішень для тренувальних даних
tree_model = DecisionTreeClassifier(random_state=123, max_depth=2)
tree_model.fit(X_train, y_train)

# Перевірка точності на тестових даних для початкової моделі
initial_pred = tree_model.predict(X_test)
initial_accuracy = accuracy_score(y_test, initial_pred)
print(f"Точність початкової моделі: {initial_accuracy:.4f}")

# Перевірка помилок на тестових даних для різних значень 'ccp_alpha' (еквівалент CP у R)
path = tree_model.cost_complexity_pruning_path(X_train, y_train)  # Шлях складності
ccp_alphas = path.ccp_alphas  # Усі значення 'ccp_alpha'
clfs = []  # Список моделей для кожного значення 'ccp_alpha'

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=123, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Перевірка помилок на тестових даних для кожного значення 'ccp_alpha'
test_errors = [1 - accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
print("Помилки на тестових даних при різних значеннях ccp_alpha:")
print(pd.DataFrame({'ccp_alpha': ccp_alphas, 'Test_Error': test_errors}))

# Знаходимо найкраще значення 'ccp_alpha', яке дає мінімальну помилку
best_index = np.argmin(test_errors)
best_ccp_alpha = ccp_alphas[best_index]
print(f"Оптимальне значення ccp_alpha для тестових даних: {best_ccp_alpha}")

# Створюємо обрізане дерево з найкращим значенням 'ccp_alpha'
best_pruned_tree = DecisionTreeClassifier(random_state=123, ccp_alpha=best_ccp_alpha)
best_pruned_tree.fit(X_train, y_train)

# Візуалізація дерева рішень
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=tree_model.classes_, filled=True, rounded=True)
plt.title("Спрощене дерево рішень для набору даних spam7 (max_depth=2)")
plt.show()

# Обчислення точності для оптимальної моделі
best_pred = best_pruned_tree.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print(f"Точність оптимальної моделі: {best_accuracy:.4f}")

# Друк точності для кожного значення 'ccp_alpha'
accuracy_results = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
results_df = pd.DataFrame({'ccp_alpha': ccp_alphas, 'Test_Accuracy': accuracy_results})
print("Точність для кожної обрізаної моделі:")
print(results_df)

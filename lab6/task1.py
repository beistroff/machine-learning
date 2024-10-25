import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Завантаження набору даних
data = pd.read_csv('data/monica.csv')

# Перегляд структури даних (інформація про кількість записів, типи змінних тощо)
print(data.info())

# Вибір ознак (X) і цільової змінної (y)
X = data.drop(columns='outcome')  # Вибираємо всі колонки, окрім 'outcome'
y = data['outcome']  # Цільова змінна 'outcome'

# Визначення категорійних і числових ознак
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Трансформер для категорійних ознак (перетворюємо категорійні змінні у набір бінарні змінні )
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # Використовуємо OneHotEncoder для перетворення категорійних ознак
    ],
    remainder='passthrough'
)

# Створення моделі дерева рішень з обмеженою глибиною, щоб уникнути перенавчання
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Спочатку застосовуємо трансформацію до категорійних змінних
    ('classifier', DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42))  # Вибираємо критерій Джині та обмежуємо глибину дерева
])

# Навчання моделі
model.fit(X, y)

# Візуалізація дерева рішень
plt.figure(figsize=(12, 8))  # Задаємо розмір графіка
plot_tree(model.named_steps['classifier'], feature_names=model.named_steps['preprocessor'].get_feature_names_out(), 
          class_names=model.named_steps['classifier'].classes_, filled=True, rounded=True)
plt.title("Спрощене дерево рішень для набору даних monica")
plt.show()

# Прогнозування результатів на основі моделі
predictions = model.predict(X)

# Виведення матриці плутанини (Confusion Matrix)
conf_matrix = confusion_matrix(y, predictions)
print("Матриця плутанини:\n", conf_matrix)

# Обчислення точності моделі
accuracy = accuracy_score(y, predictions)
print(f"Точність моделі: {accuracy:.4f}")

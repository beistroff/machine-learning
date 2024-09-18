import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

print(data.head())
# data_head = data.head()
# data_head.index += 1
# print(tabulate(data_head, headers='keys', tablefmt='grid', showindex=True))

X = data.drop('Outcome', axis=1)  # Всі ознаки
y = data['Outcome']  # Цільова змінна (0 або 1 - наявність чи відсутність діабету)

# Поділ на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Створення моделі наївного баєсівського класифікатора
nb_classifier = GaussianNB()

# Тренування моделі на тренувальних даних
nb_classifier.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = nb_classifier.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

table = [["", "Передбачено: Немає діабету", "Передбачено: Є діабет"],
         ["Фактично: Немає діабету", conf_matrix[0][0], conf_matrix[0][1]],
         ["Фактично: Є діабет", conf_matrix[1][0], conf_matrix[1][1]]]

report = classification_report(y_test, y_pred)


print(f"\n\033[92mТочність моделі: {accuracy:.4f}\033[0m")
print("\nМатриця плутанини:")
print(tabulate(table, headers="firstrow", tablefmt="grid"))
print("\nЗвіт по класифікації:")
print(report)

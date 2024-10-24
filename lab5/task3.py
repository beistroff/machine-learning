import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def task3():
    train_data = pd.read_csv('data/svmdata3.txt', delimiter='\t')
    test_data = pd.read_csv('data/svmdata3test.txt', delimiter='\t')

    # Розділення на ознаки та цільову змінну
    X_train = train_data[['X1', 'X2']]
    y_train = train_data['Colors']
    X_test = test_data[['X1', 'X2']]
    y_test = test_data['Colors']

    # Перетворення категоріальних значень на числові
    y_train = y_train.map({'red': 0, 'green': 1})
    y_test = y_test.map({'red': 0, 'green': 1})

    # Ініціалізація результатів
    results = {}

    # Функція для тренування та оцінки моделі
    def evaluate_svm(kernel, **kwargs):
        clf = SVC(kernel=kernel, **kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return 1 - accuracy  # Помилка

    # Поліноміальне ядро з різними degree
    for degree in [2, 3, 4]:
        error = evaluate_svm('poly', degree=degree)
        results[f'polynomial_degree_{degree}'] = error

    # Радіальне ядро (RBF)
    error = evaluate_svm('rbf')
    results['radial'] = error

    # Сигмоїдне ядро
    error = evaluate_svm('sigmoid')
    results['sigmoid'] = error

    print("Помилки на тестовій вибірці для кожного ядра:")
    for kernel, error in results.items():
        print(f'{kernel}: {error:.4f}')

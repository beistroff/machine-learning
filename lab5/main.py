import task1
import task2
import task3
import task4

def main():
    while True:
        print("\n--- Виберіть задачу ---")
        print("1. Завдання 1. Побудова машини опорних векторів C=1")
        print("2. Завдання 2. Досягнення нульової помилки класифікації")
        print("3. Завдання 3. Вибір оптималього ядра")
        print("4. Завдання 4. Побудова машини опорних векторів типу 'eps-regression'")
        print("5. Вийти")

        choice = input("Ваш вибір: ")

        if choice == '1':
            print("\033[92m\nВиконання завдання 1. \033[0m")
            task1.task1()
        elif choice == '2':
            print("\033[92m\nВиконання завдання 2. \033[0m")
            task2.task2()
        elif choice == '3':
            print("\033[92m\nВиконання завдання 3: \033[0m")
            task3.task3()
        elif choice == '4':
            print("\033[92m\nВиконання завдання 4: \033[0m")
            task4.task4()
        elif choice == '5':
            print("Вихід.")
            break
        else:
            print("Не правильний вибір, спробуйте ще раз.")

if __name__ == "__main__":
    main()

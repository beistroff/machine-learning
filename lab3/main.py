import task1
import task2
import task3
import task4
import task5

def main():
    while True:
        print("\n--- Виберіть задачу ---")
        print("1. Завдання 1. Використання ф-ї lm() та дані з файлу  - reglab1.txt")
        print("2. Завдання 2. Алгоритм для зменшення кількості ознак - reglab2.txt")
        print("3. Завдання 3. Побудова регресії з файлу - cygage.txt")
        print("4. Завдання 4. Найкраща модель - alligators.txt")
        print("5. Завдання 5. Гребнева регресія, dataset longley")
        print("6. Вийти")

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
            print("\033[92m\nВиконання завдання 5: \033[0m")
            task5.task5()
        elif choice == '6':
            print("Вихід.")
            break
        else:
            print("Не правильний вибір, спробуйте ще раз.")

if __name__ == "__main__":
    main()

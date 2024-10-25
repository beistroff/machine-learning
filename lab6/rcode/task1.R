library(DAAG)
# Для побудови дерева класифікації та візуалізації
library(rpart)
library(rpart.plot)

data(monica)
str(monica)

# Побудова дерева за індексом Джині (за замовчуванням)
tree_model_gini <- rpart(outcome ~ ., data = monica, method = "class")

# Візуалізація дерева з показниками Джині та додатковою інформацією
rpart.plot(tree_model_gini, type = 4, extra = 106, under = TRUE, faclen = 0,
           fallen.leaves = TRUE, varlen = 0, digits = 3,
           main = "Дерево класифікації з індексом Джині для набору даних monica")

# Перегляд результатів моделі
summary(tree_model_gini)

# Прогнозування на основі моделі (клас, який має найбільшу ймовірність)
predictions <- predict(tree_model_gini, monica, type = "class")

# Перегляд матриці плутанини (confusion matrix)
conf_matrix <- table(monica$outcome, predictions)
print(conf_matrix)

# Обчислення точності моделі (accuracy)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Точність моделі:", round(accuracy, 4)))

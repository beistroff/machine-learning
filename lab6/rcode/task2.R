library(DAAG)
library(tree)
library(e1071)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)

data(spam7)

# 70% тренувальних, 30% тестових
set.seed(123)
train_index <- createDataPartition(spam7$yesno, p = 0.7, list = FALSE)
train_data <- spam7[train_index, ]
test_data <- spam7[-train_index, ]

# Побудова регресійного дерева
tree_model <- rpart(yesno ~ ., data = train_data, method = "class")

# Виведення дерева
rpart.plot(tree_model, type = 3, extra = 104, under = TRUE, faclen = 0,
           main = "Регресійне дерево для набору даних spam7")

# Виведення графіку складності (CP)
printcp(tree_model)
plotcp(tree_model)

# Перевірка помилок на тестових даних для різних значень CP
error_test <- sapply(tree_model$cptable[, "CP"], function(cp) {
  pruned <- prune(tree_model, cp = cp)
  pred <- predict(pruned, test_data, type = "class")
  return(1 - sum(pred == test_data$yesno) / nrow(test_data))
})

# Друк помилок для різних значень CP
cat("Помилки на тестових даних при різних значеннях CP:\n")
print(data.frame(CP = tree_model$cptable[, "CP"], Test_Error = error_test))

# Обчислення точності для початкової моделі
initial_pred <- predict(tree_model, test_data, type = "class")
initial_accuracy <- sum(initial_pred == test_data$yesno) / nrow(test_data)
cat("Точність початкової моделі:", initial_accuracy, "\n")

# Вибір найкращого значення CP за найменшою помилкою на тестових даних
best_cp <- tree_model$cptable[which.min(error_test), "CP"]
cat("Оптимальне значення CP для тестових даних:", best_cp, "\n")

# Обрізка дерева з оптимальним значенням CP
best_pruned_tree <- prune(tree_model, cp = best_cp)

# Візуалізація оптимального обрізаного дерева
rpart.plot(best_pruned_tree, type = 3, extra = 104, under = TRUE, faclen = 0,
           main = "Оптимальне обрізане дерево для набору даних spam7")

# Обчислення точності для кожної обрізаної моделі
cat("Точність для кожної обрізаної моделі:\n")
accuracy_results <- sapply(tree_model$cptable[, "CP"], function(cp) {
  pruned <- prune(tree_model, cp = cp)
  pred <- predict(pruned, test_data, type = "class")
  return(sum(pred == test_data$yesno) / nrow(test_data))  # Обчислюємо точність
})

# Друк точності для різних значень CP
results_df <- data.frame(CP = tree_model$cptable[, "CP"], Test_Accuracy = accuracy_results)
print(results_df)

# Візуалізація дерев для кожного значення CP
for (cp_value in tree_model$cptable[, "CP"]) {
  pruned_tree_cp <- prune(tree_model, cp = cp_value)
  cat("\nДерево для CP =", cp_value, "\n")
  rpart.plot(pruned_tree_cp, type = 3, extra = 104, under = TRUE, faclen = 0,
             main = paste("Дерево для CP =", cp_value))
}

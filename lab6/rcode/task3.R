library(DAAG)
library(tree)
library(e1071)
library(caret)
library(ggplot2)

data(nsw74psid1)

# Видаляємо рядки з пропущеними значеннями
nsw74psid1 <- na.omit(nsw74psid1)

# Функція для мінімум-максимум нормалізації
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Нормалізація всіх числових змінних в наборі даних
nsw74psid1_normalized <- as.data.frame(lapply(nsw74psid1, function(col) {
  if(is.numeric(col)) return(normalize(col))
  else return(col)
}))

# Оцінка моделі за допомогою перехресної перевірки
set.seed(123)
train_index <- createDataPartition(nsw74psid1_normalized$re78, p = 0.8, list = FALSE)
train_data <- nsw74psid1_normalized[train_index, ]
test_data <- nsw74psid1_normalized[-train_index, ]

# Побудова регресійного дерева для re78 ~ .
tree_model <- tree(re78 ~ ., data = train_data)

# Виведення та візуалізація початкового дерева
plot(tree_model)
text(tree_model, pretty = 0)

# Прогнозування за допомогою дерева
tree_predictions <- predict(tree_model, newdata = test_data)

# Обчислення RMSE для дерева
tree_rmse <- sqrt(mean((tree_predictions - test_data$re78)^2))

# Побудова SVM моделі
svm_model <- svm(re78 ~ ., data = train_data)

# Прогнозування за допомогою SVM
svm_predictions <- predict(svm_model, newdata = test_data)

# Обчислення RMSE для SVM
svm_rmse <- sqrt(mean((svm_predictions - test_data$re78)^2))

# Виведення результатів
cat("RMSE для SVM моделі:", svm_rmse, "\n")
cat("RMSE для регресійного дерева:", tree_rmse, "\n")

# Візуалізація результатів SVM
results <- data.frame(Real = test_data$re78, Predicted = svm_predictions)

ggplot(results, aes(x = Real, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(title = "Прогнозування SVM: Реальні vs Прогнозовані значення",
       x = "Реальні значення",
       y = "Прогнозовані значення") +
  theme_minimal()

if (tree_rmse < svm_rmse) {
  cat("Регресійне дерево має кращу якість моделі.\n")
} else {
  cat("SVM має кращу якість моделі.\n")
}
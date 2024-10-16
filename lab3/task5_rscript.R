# Підключення бібліотеки glmnet для використання алгоритмів регресії з регуляризацією.
library(glmnet)
cat("\014") 

data(longley)
longley <- longley[, !names(longley) %in% "Population"] # видалення колонки Population

n <- nrow(longley)
train_indices <- sample(1:n, n/2) # 50/50 %
train_data <- longley[train_indices, ]
test_data <- longley[-train_indices, ]

# Незалежні змінні
train_x <- as.matrix(train_data[, -ncol(train_data)])
test_x <- as.matrix(test_data[, -ncol(test_data)])

# Залежні змінні
train_y <- train_data[["Unemployed"]]
test_y <- test_data[["Unemployed"]]

# Генерація значень λ для гребневої регресії.
lambda_values <- 10^(-3 + 0.2 * (0:25))

# alpha = 0 вказує на те, що ми побудуємо гребеневу регресію (ridge regression)
ridge_model <- glmnet(train_x, train_y, alpha = 0, lambda = lambda_values)

# Обчислення помилок для кожного значення λ.
train_errors <- sapply(1:length(lambda_values), function(i) {
  preds <- predict(ridge_model, s = lambda_values[i], newx = train_x)

  # Обчислення середньої квадратичної помилки.
  mean((train_y - preds)^2)
})

test_errors <- sapply(1:length(lambda_values), function(i) {
  preds <- predict(ridge_model, s = lambda_values[i], newx = test_x)
  mean((test_y - preds)^2)
})

plot(lambda_values, train_errors, type = "b", col = "yellow", 
     xlab = "Лямбда", ylab = "Середня квадратична помилка", 
     main = "Помилка на навчальній і тестовій вибірках", 
     xlim = c(0, 20))
lines(lambda_values, test_errors, type = "b", col = "blue")  # тестові помилки
legend("topright", legend = c("Train Error", "Test Error"), col = c("yellow", "blue"), lty = 1)  

# Знаходження індексу мінімальної помилки на тестовій вибірці
min_lambda_index <- which.min(test_errors)

# Виведення відповідного значення λ
best_lambda <- lambda_values[min_lambda_index]
cat("Значення λ з найменшою помилкою:", best_lambda, "\n")

cat("Найменша помилка на тестовій вибірці:", test_errors[min_lambda_index], "\n")

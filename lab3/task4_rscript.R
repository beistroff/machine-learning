library(caret)
library(ggplot2)
library(MASS)

cat("\014")

data <- read.table("D:/university/4d_course/1st_term/machine-learning/lab3/data/ALIGATORS.txt", header = TRUE)

split <- createDataPartition(data$Weight, p = 0.7, list = FALSE)
train_data <- data[split, ]
test_data <- data[-split, ]

# Лінійна модель
model_linear <- lm(Weight ~ Length, data = train_data)

# Поліноміальні моделі 2-го, 3-го, 4-го та 5-го порядку.
model_poly2 <- lm(Weight ~ poly(Length, 2), data = train_data)
model_poly3 <- lm(Weight ~ poly(Length, 3), data = train_data)
model_poly4 <- lm(Weight ~ poly(Length, 4), data = train_data)
model_poly5 <- lm(Weight ~ poly(Length, 5), data = train_data)

models <- list(
  linear = model_linear,
  poly2 = model_poly2,
  poly3 = model_poly3,
  poly4 = model_poly4,
  poly5 = model_poly5
)

calculate_rss_and_mse <- function(model, test_data) {
  predictions <- predict(model, newdata = test_data)
  true_values <- test_data$Weight
  squared_differences <- (true_values - predictions)^2
  rss <- sum(squared_differences)
  mse <- mean(squared_differences)
  return(list(rss = rss, mse = mse))
}

results <- data.frame(Model = character(), RSS = numeric(), MSE = numeric(), stringsAsFactors = FALSE)

for (model_name in names(models)) {
  model <- models[[model_name]]
  metrics <- calculate_rss_and_mse(model, test_data)
  results <- rbind(results, data.frame(Model = model_name, RSS = metrics$rss, MSE = metrics$mse, stringsAsFactors = FALSE))
}

cat("\nЗалишкова сума квадратів (RSS) та середньо-квадратична похибка (MSE) для кожної моделі:\n")
print(results)

best_model_name <- results$Model[which.min(results$RSS)]
best_model <- models[[best_model_name]]

cat("\nНайкраща модель за RSS:", best_model_name, "\n")
cat("\nПідсумковий звіт моделі:\n")
print(summary(best_model))

ggplot() +
  geom_point(data = data, aes(x = Length, y = Weight), color = 'blue', alpha = 0.5) +
  geom_smooth(data = train_data, aes(x = Length, y = Weight, color = 'Linear'), method = 'lm', formula = y ~ x, se = FALSE) + 
  geom_smooth(data = train_data, aes(x = Length, y = Weight, color = 'Poly 2'), method = 'lm', formula = y ~ poly(x, 2), se = FALSE) + 
  geom_smooth(data = train_data, aes(x = Length, y = Weight, color = 'Poly 3'), method = 'lm', formula = y ~ poly(x, 3), se = FALSE) + 
  geom_smooth(data = train_data, aes(x = Length, y = Weight, color = 'Poly 4'), method = 'lm', formula = y ~ poly(x, 4), se = FALSE) + 
  geom_smooth(data = train_data, aes(x = Length, y = Weight, color = 'Poly 5'), method = 'lm', formula = y ~ poly(x, 5), se = FALSE) + 
  geom_point(data = test_data, aes(x = Length, y = Weight, color = 'Test Data'), shape = 1) +
  geom_line(data = test_data, aes(x = Length, y = predict(best_model, newdata = test_data), color = 'Predictions'), linetype = 'dashed') +
  labs(title = paste('Модель для залежності ваги від довжини (', best_model_name, ')', sep = ''),
       x = 'Довжина',
       y = 'Вага') +
  scale_color_manual(
    name = 'Model', 
    values = c(
        'Linear' = 'black', 
        'Poly 2' = 'blue', 
        'Poly 3' = 'orange', 
        'Poly 4' = 'green', 
        'Poly 5' = 'red', 
        'Test Data' = 'red', 
        'Predictions' = 'purple')
    ) + theme_minimal() 
      + coord_cartesian(ylim = c(min(data$Weight), max(data$Weight + 1)))

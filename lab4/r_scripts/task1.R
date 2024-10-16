cat("\014")

# Завантажуємо dataset iris
data(iris)

# Встановлюємо рівень значущості
alpha = 0.05

# Виконуємо тест Шапіро для кожного стовпця ознак
for (column in colnames(iris)[1:4]) {
  result <- shapiro.test(iris[[column]])
  cat("Характеристика:", column, "\n")
  cat("p-value:", result$p.value, "\n")
  cat("Статистика W:", result$statistic, "\n")
  
  if(result$p.value > alpha){
    cat("p-значення більше за", alpha, ", тому розподіл", column, "нормальний.\n")
  } else {
    cat("p-значення менше або дорівнює", alpha, ", тому розподіл", column, "не нормальний.\n\n")
  }
}

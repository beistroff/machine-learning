cat("\014")

women_data <- HairEyeColor[,, "Female"]

chi_test <- chisq.test(women_data)
cat("Статистика X^2: ", chi_test$statistic, "\n")
cat("p-value:", chi_test$p.value, "\n")
cat("Ступені свободи: ", chi_test$parameter, "\n")
cat("Очікувані значення:","\n")

chi_test$expected

# Перевірка гіпотези
if(chi_test$p.value > 0.05){
  print("Кольори очей і волосся не залежать один від одного.")
} else{
  print("Кольори очей і волосся залежать один від одного.")
}
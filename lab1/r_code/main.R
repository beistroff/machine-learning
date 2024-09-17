install.packages(c("psych", "dplyr", "ggplot2"))
library("psych")
library("ggplot2")
library("dplyr")

#--------------------------1--------------------------------------
d<-iris
describe(d)

summary(d)
head(d)

ggplot(iris, aes(x=Petal.Length, y=Petal.Width, color=Species)) +
  geom_point() +
  labs(title="Розподіл довжини та ширини пелюсток для кожного виду Iris")


#--------------------------2--------------------------------------
sepal_length <- iris$Sepal.Length

# Обчислення основних вибіркових характеристик
mean_value <- mean(sepal_length)
variance_value <- var(sepal_length) 
mode_value <- as.numeric(names(sort(table(sepal_length), decreasing=TRUE)[1]))
median_value <- median(sepal_length)
sd_value <- sd(sepal_length)

# Print results
cat("Середнє:", mean_value, "\n")
cat("Дисперсія:", variance_value, "\n")
cat("Мода:", mode_value, "\n")
cat("Медіана:", median_value, "\n")
cat("Стандартне відхилення:", sd_value, "\n")


#--------------------------3--------------------------------------

# Гістограма щільності розподілу
x <- seq(3,8,by=0.1)

hist(sepal_length, probability=TRUE, main="Гістограма щільності розподілу Sepal.Length", 
     xlab="Sepal Length", col="grey", border="black")
lines(x, dnorm(x, mean_value, variance_value), type='l', col='red')

# Гістограма частот
p <- qplot(sepal_length)
p + geom_hline(yintercept = mean_value, linetype = "dashed", color = "blue")


#--------------------------4--------------------------------------
# Створення вектора
vector_result <- ifelse(sepal_length > mean_value, 1, 
                        ifelse(sepal_length < mean_value, -1, 0))

# Виведення вектора
print(vector_result)


#--------------------------5--------------------------------------
# Генерація вибірки з нормального розподілу
d3 <- rnorm(250, mean(sepal_length), var(sepal_length))

# Побудова графіка емпіричної функції розподілу
plot(ecdf(d3), main="Емпірична функція розподілу", 
     xlab="Значення", ylab="Ймовірність", col="black")

# Графік функції розподілу
x<-seq(3,8,by=0.1)
lines(x, pnorm(x,mean(sepal_length),var(sepal_length)), type='l', col='red')


#--------------------------6--------------------------------------
d3_mean <- mean(d3)
d3_sd <- sd(d3)

lines(x, pnorm(x, d3_mean, d3_sd), type='l', col='blue')


#--------------------------7--------------------------------------
# Графік щільності розподілу
x <- seq(0, 10, length=100)

# Щільність для N(0, 1) нормамального розподілу
plot(x, dnorm(x, mean=0, sd=1), type="l", col="blue", ylim=c(0, 0.5), 
     ylab="Щільність", xlab="Значення", main="Графіки щільності розподілу")

# Щільність для N(0, 2)
lines(x, dnorm(x, mean=0, sd=2), col="red")

# Щільність для N(1, 1)
lines(x, dnorm(x, mean=1, sd=1), col="green")


#--------------------------8--------------------------------------
set.seed(123)
Nrow <- 1:20
Name <- paste("Employee", Nrow)
BirthYear <- sample(1970:1990, 20, replace=TRUE)
EmployYear <- sapply(BirthYear, function(x) sample((x+18):2019, 1))
Salary <- (log(2020 - EmployYear) + 1) * 10000

# Створення data frame
employees <- data.frame(Nrow, Name, BirthYear, EmployYear, Salary)

# Підрахунок кількості співробітників з зарплатою понад 20000
num_high_salary <- sum(employees$Salary > 20000)

# Додавання поля для податку на прибуток
employees$Tax <- (2020 - employees$EmployYear) * 0.13 * employees$Salary

# Виведення результатів
print(employees)
cat("Кількість співробітників із зарплатою понад 20000:", num_high_salary, "\n")

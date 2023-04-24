setwd("~/Documents/1-UNC/1-Classes/BIOS663/Final Project")

library(readr)
library(ggplot2)
library(tidyverse)
library(MASS, include.only = c("boxcox"))
library(car, include.only = c("vif"))
library(knitr)

# Read in data and set chas to factor
boston <- read_table2("boston.txt", col_names = FALSE)
boston <- data.frame(boston)
error_indices <- seq(2, 1012, 2)
last_3_cols <- boston[error_indices, 1:3]
boston <- boston[-error_indices,]
boston <- cbind(boston, last_3_cols)
names(boston) <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis",
                   "rad", "tax", "ptratio", "b", "lstat", "medv")
rownames(boston) <- NULL # reset indices
boston$chas <- factor(boston$chas)
write.csv(boston, "boston.csv", row.names=FALSE)

# Split into training and testing: 80-20 split
set.seed(123)
trainset <- sample(seq(1:nrow(boston)), size=0.8*nrow(boston), replace=FALSE)
train <- boston[trainset,]
test <- boston[-trainset,]
rownames(train) <- NULL
rownames(test) <- NULL
write.csv(train, "train.csv", row.names=FALSE)
write.csv(test, "test.csv", row.names=FALSE)

# Bivariate scatter plots to determine transformations
plot(train$crim, train$medv)
plot(log(train$crim+1), train$medv) # transformation for crime makes relationship more linear, but with outliers
plot(train$zn, train$medv)
plot(train$indus, train$medv)
plot(train$chas, train$medv)
plot(train$nox, train$medv)
plot(train$rm, train$medv)
plot(train$age, train$medv)
plot(train$dis, train$medv)
plot(log(train$dis), train$medv) # log transform on dis clearly makes relationship more linear
plot(train$rad, train$medv)
plot(log(train$rad), train$medv) # log transform helps spread values more equally over x axis
plot(train$tax, train$medv)
plot(log(train$tax), train$medv)
plot(train$b, train$medv)
plot(train$lstat, train$medv)
plot(log(train$lstat), train$medv) # log transform on lstat clearly makes relationship more linear

# Nice plot showing fitted curve after log transform on lstat
ggplot(train, aes(x=lstat, y=medv)) + 
  geom_point() +
  stat_smooth(method='lm', formula = y ~ log(x), linewidth = 1) + 
  xlab('lstat') +
  ylab('medv')

# Box-cox transformation on full model with untransformed response variable
bxcx <- boxcox(lm(medv ~ log(1+crim) + zn + indus + chas + nox + rm + age + log(dis) + 
                    log(rad) + log(tax) + ptratio + b + log(lstat), 
                  data = train)) # consider using train[-c(341, 218, 352),] here
lambda <- bxcx$x[which.max(bxcx$y)]
lambda

# Full model
full.model <- lm(medv^lambda ~ log(1+crim) + zn + indus + chas + nox + rm + age + log(dis) + 
                   log(rad) + log(tax) + ptratio + b + log(lstat), 
                 data = train)

summary(full.model)
confint(full.model)

# Default diagnostics plots
par(mfrow = c(2, 2))
plot(full.model)
par(mfrow = c(1, 1))
# Residuals appear slightly light tailed, but overall no major violations of assumptions

# Cook's Distance
plot(full.model, 4)

# Histogram of residuals; looks reasonably normal with some outliers
hist(full.model$residuals, breaks = 25)

## Collinearity analysis ##
# Variance inflation factors
vif(full.model)

# Eigenanalysis: scaled SSCP matrix
X <- model.matrix(full.model)
Ds_inv_sqrt <- diag((diag(t(X) %*% X))^(-0.5))
SSCP_scaled <- Ds_inv_sqrt %*% (t(X) %*% X) %*% Ds_inv_sqrt
eigenvalues <- eigen(SSCP_scaled)$values
condition <- sqrt(eigenvalues[1]/eigenvalues)
eigenanalysis <- cbind(round(eigenvalues, 4), condition)
colnames(eigenanalysis) <- c("Eigenvalue", "Condition Index")
kable(eigenanalysis, format = "pipe", caption = "Eigenanalysis of scaled SSCP matrix")
eigen(SSCP_scaled)$vectors[,14] # first and eleventh values clearly nonzero, indicating issues with the intercept and log(tax)

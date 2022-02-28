# Week 5 course assignment

library(dplyr)
library(car)
library(fitdistrplus)

# Loading the data
setwd(...)
data <- read.csv("Outsourcing_case_raw_data.csv", sep=";")
View(data)

# Setting up a summary table
data_agg <- data %>%
  group_by(LEC.ID.) %>%
  summarize(sum_sales = sum(Units.Sold), duration = n(), sales_per_day = sum(Units.Sold)/duration, holiday = if(LEC.ID. == 18 | LEC.ID. == 19) 1 else 0 )

View(data_agg)
plot(data_agg$sum_sales)
hist(data_agg$sum_sales)

# Choosing a regression model

# Model 1, DV: Average sales on a single day
reg <- lm(sales_per_day ~ duration + holiday, data_agg)
summary(reg)

# Model 2, DV: Sum of sales
reg2 <- lm(sum_sales ~ duration + holiday, data_agg)
summary(reg2)
vif(reg2) # No multicollinearity observed

# Model 2 yields better results, so we chose to use it for our predictions
data_agg$prediction = predict(reg2)
data_agg$residual = data_agg$sum_sales - data_agg$prediction
summary(data_agg$residual)

# Residuals
plot(data_agg$residual, ylab="Residuals", main="Residual plot")
abline(h=0)
hist(data_agg$residual, xlab="Residuals", main="Histogram of residuals")

# Fitting normal distribution to residuals
resid_norm_fit <- fitdist(data_agg$residual, "norm")
summary(resid_norm_fit)
plot(resid_norm_fit)

# Based on the Q-Q plot it seems that normal distribution is a rather good assumption regarding our data

# Actual-to-forecast ratios
a2f_ratios <- data_agg$sum_sales/data_agg$prediction #Create the Actual to Forecast ratio variable
hist(a2f_ratios) 

a2f_norm_fit <- fitdist(a2f_ratios, "norm")
a2f_lnorm_fit <- fitdist(a2f_ratios, "lnorm")
a2f_gamma_fit <- fitdist(a2f_ratios, "gamma")

cdfcomp(list(a2f_norm_fit, a2f_lnorm_fit, a2f_gamma_fit))
gofstat(list(a2f_norm_fit, a2f_lnorm_fit, a2f_gamma_fit))
# Minimum AIC & BIC value suggest that either lnorm or gamma distribution would be the best assumption for the underlying distribution of our data

# Modeling the demand
n <- 100
resid_means_boot <-  rep(0, n) #Output variable
for(i in 1:n) {
  resid_sample <- sample(data_agg$residual, 100, replace = TRUE) #Sampling is performed with replacement
  resid_means_boot[i] <- mean(resid_sample) #calculate the mean for the sample
}
summary(resid_means_boot) #Summary of the bootstrapped mean of residual distribution.
sd(resid_means_boot)

# Modeling the demand
dur <- 21
demand_pred <- reg2$coefficients[1] + reg2$coefficients[2]*dur
demand_rand_observations <- rnorm(100, demand_pred, sd(data_agg$residual))

# See excel for further analysis.
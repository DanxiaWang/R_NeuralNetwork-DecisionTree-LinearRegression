
save.image("FinalProject.Rdata")
load("FinalProject.Rdata")

library(zoo)
library(sandwich)
library(lmtest)

# load data
insurance
insurance <- read.csv("/Users/danxiawang/Documents/R/Project/insurance.csv", header = TRUE) 
insurance.df <- data.frame(insurance)
View(insurance.df)
dim(insurance.df)
head(insurance.df)

is.null(insurance.df)
summary(insurance.df)

# change smoker to dummy variable
insurance.df$smoker <- ifelse(insurance.df$smoker=="yes",1,0)
insurance.df$smoker

insurance.df$sex <- ifelse(insurance.df$sex=="male",1,0)
insurance.df$sex



# Pair all the regressor
library(magrittr)
library(dplyr)
library(GGally)
insurance.df %>% 
        select(age,sex, bmi, children, smoker, region, charges) %>%
        GGally::ggpairs(mapping = aes(color = region))
# we can see all the plot between charges and age, bmi, smoker, region, sex and children

cor(insurance.df$smoker, insurance.df$bmi)

cor(insurance.df$charges,insurance.df$smoker)
cor(insurance.df$charges,insurance.df$age)
cor(insurance.df$charges,insurance.df$bmi)
cor(insurance.df$charges,insurance.df$children)
cor(insurance.df$charges,insurance.df$sex)

# ggplot the charges on bmi for smoker
library(ggplot2)
ggplot(insurance.df, aes(x=bmi, y=charges, color=smoker)) + geom_point(size=3)+
        scale_color_gradient(low="cyan4", high="coral1")
        
# For Smoker, the charges increases clearly

# Regression of charges on bmi
insurance_1 <- lm(charges~bmi, data = insurance.df)
coeftest(insurance_1, vcov. = vcovHC, type = "HC1")
summary(insurance_1)

# Test diagnostic
par(mfrow=c(2,2))
plot(charge_bmi)
dev.off()

# Regression of charges on bmi
insurance_1 <- lm(charges~bmi, data = insurance.df)
coeftest(insurance_1, vcov. = vcovHC, type = "HC1")
summary(insurance_1)

# cubic model
cubic_model <- lm(charges~poly(bmi, degree = 3, raw = TRUE), data = insurance.df) 
coeftest(cubic_model, vcov. = vcovHC, type = "HC1")
summary(cubic_model)

cubic_model_1 <- lm(charges~poly(bmi, degree = 3, raw = TRUE)+smoker+age+children, data = insurance.df) 
coeftest(cubic_model_1, vcov. = vcovHC, type = "HC1")
summary(cubic_model_1)

qudredic <- lm(charges~bmi+I(bmi^2), data = insurance.df)
summary(qudredic)

# set up hypothesis matrix
R <- rbind(c(0, 0, 1, 0),
           c(0, 0, 0, 1))

# perform robust F-test
library(car)
linearHypothesis(cubic_model, 
                 hypothesis.matrix = R,
                 vcov. = vcovHC, type = "HC1")



# Regression for charges on bmi+smoker
insurance_2 <- lm(charges~bmi+smoker, data = insurance.df)
coeftest(insurance_2, vcov. = vcovHC, type = "HC1")
summary(insurance_2)

# charges on bmi*smoker
insurance_3 <- lm(charges~bmi+smoker+bmi*smoker, data = insurance.df)
summary(insurance_3)

# Regression for charges on bmi+smoker+age
insurance_4 <- lm(charges~bmi+smoker+age, data = insurance)
coeftest(insurance_4, vcov. = vcovHC, type = "HC1")
summary(insurance_4)

# Regression for charges on bmi+smoker+bmi*smoker+age
insurance_5 <- lm(charges~bmi+smoker+bmi*smoker+age, data = insurance)
coeftest(insurance_5, vcov. = vcovHC, type = "HC1")
summary(insurance_5)

# charges on log(bmi)
insurance_6 <- lm(charges~log(bmi), data = insurance.df)
summary(insurance_6)

# charges on log(bmi)+smoker
insurance_7 <- lm(charges~log(bmi)+smoker, data = insurance.df)
summary(insurance_7)

# charges on log(bmi)+smoker+age
insurance_8 <- lm(charges~log(bmi)+smoker+age, data = insurance.df)
summary(insurance_8)



# gather clustered standard errors in a list
rob_se <- list(sqrt(diag(vcovHC(insurance_1, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_2, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_3, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_4, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_5, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_6, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_7, type = "HC1"))),
               sqrt(diag(vcovHC(insurance_8, type = "HC1"))))

library(stargazer)
stargazer(insurance_1,insurance_2,insurance_3,insurance_4,insurance_5,insurance_6,insurance_7,insurance_8, 
          type="html",
          digits = 3,
          se = rob_se,
          title = "Regression Models of Insurance Charges due to BMI",
          out="Insurance_model.htm")




#Part 1: Data Visualisation 

plot(charges~age,
     data = insurance,
     col  = "steelblue",
     xlab = "Age",
     ylab = "Insurance Charges (U.S. Dollars)",
     main = "Health Insurance Payments and Age Plot")

cor(insurance$charges,insurance$age)

plot(charges~children, data = insurance.df)


plot(charges~bmi,
     data = insurance.df,
     col  = "steelblue",
     xlab = "BMI Score",
     ylab = "Insurance Charges (U.S. Dollars)",
     main = "Health Insurance Payments and BMI Score Plot")

cor(insurance$charges,insurance$bmi)

#Create dummy variable for "smoker" variable:

boxplot(charges~smoker.bin,
     data = insurance.df,
     col  = "steelblue",
     xlab = "Smoker (Yes=1; No=0)",
     ylab = "Insurance Charges (U.S. Dollars)",
     main = "Health Insurance Payments for Smokers vs Non-Smokers BoxPlot")

cor(insurance.df$charges,insurance.df$smoker.bin)

#Create dummy variable for "region" variable:
insurance$region_count<-ifelse(insurance$region=="southwest",1,
                               ifelse(insurance$region=="southeast",2,
                                      ifelse(insurance$region=="northwest",3,
                                             ifelse(insurance$region=="northeast",4,1))))
View(insurance)

insurance.df$southwest<-ifelse(insurance.df$region=="southwest",1,0)
View(insurance.df)

insurance.df$southeast<-ifelse(insurance.df$region=="southeast",1,0)
View(insurance.df)

insurance.df$northwest<-ifelse(insurance.df$region=="northwest",1,0)
View(insurance.df)

insurance.df$northeast<-ifelse(insurance.df$region=="northeast",1,0)
View(insurance.df)

boxplot(charges~region.count,
        data = insurance.df,
        col  = "steelblue",
        xlab = "Region",
        ylab = "Insurance Charges (U.S. Dollars)",
        main = "Health Insurance Payments for Different Regions in the U.S.")

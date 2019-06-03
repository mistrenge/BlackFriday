install.packages("tidyverse")
install.packages("caret")
install.packages("caretEnsemble")
install.packages("plyr")
install.packages("MASS")
install.packages("Hmisc")
install.packages("gclus")
install.packages("glmnet")
library(tidyverse)
library(caret)
library(caretEnsemble)
library(plyr)
library(MASS)
library(Hmisc)
library(gclus)
library(glmnet)

# Load data set BlackFriday from my GitHub account
df <- read.csv("https://raw.githubusercontent.com/mistrenge/BlackFriday/master/BlackFriday.csv")

# Explore original dataset
head(df[1:7])
head(df[8:12])
str(df)

# Select variables for further analysis
df <- df %>%
  dplyr::select(User_ID, Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status, Purchase)

# Transform format of variables to better manipulate data in next step
df <- within(df, {
  Occupation <- as.factor(Occupation)
  Marital_Status <- as.factor(Marital_Status)
})

# Summarize data set to calculate summary purchase for each customer
df <- df %>% group_by(User_ID) %>%
  summarise_each(funs(if(is.numeric(.)) sum(., na.rm = TRUE) else first(.)))

# Transform data to numeric variables
revalue(df$Gender, c("F" = 1, "M" = 0)) -> df$Gender
revalue(df$Age, c("0-17" = 0, "18-25" = 1, "26-35" = 2, "36-45"= 3, "46-50" = 4, "51-55" = 5, "55+" = 6)) -> df$Age
revalue(df$City_Category, c("C" = 0, "B" = 1, "A" = 2)) -> df$City_Category
revalue(df$Stay_In_Current_City_Years, c("0" = 0, "1" = 1, "2" = 2, "3"= 3, "4+" = 4)) -> df$Stay_In_Current_City_Years

df <- within(df, {
  Gender <- as.numeric(as.character(Gender))
  Age <- as.numeric(as.character(Age))
  City_Category <- as.numeric(as.character(City_Category))
  Stay_In_Current_City_Years <- as.numeric(as.character(Stay_In_Current_City_Years))
  Marital_Status <- as.numeric(as.character(Marital_Status))
  Occupation <- as.numeric(as.character(Occupation))
  Purchase <- as.numeric(as.character(Purchase))
})

# Check dataset
head(df)
dim(df)
str(df)

# Identify missing values
sum(is.na(df))

# Explore data set and create histogram for purchase variable
summary(df)
df %>% group_by(User_ID) %>% ggplot(aes(Purchase)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle ("Purchase per User")

# Log transform Purchase variable
df <- df %>% mutate(Purchase = log(Purchase))

# Show correlations between variables in training set
cor <-rcorr(as.matrix(df[2:8]))
cor

# Create scatter plot matrix to show correlations between variables in training set
dta <- df[c(2:8)] # get data 
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors

# Reorder variables so those with highest correlation are closest to the diagonal
dta.o <- order.single(dta.r) 
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation")

# Select variables for machine learning model
df <- df %>%
  dplyr::select(Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status, Purchase)

# Split training and test set
set.seed(1)
test_index <- createDataPartition(y = df$Purchase, times = 1, p = 0.7, list = FALSE)
train_set <- df[-test_index,]
test_set <- df[test_index,]

# Show dimensions of training set
dim(train_set)


# Step 1: build machine learning model for training data with all independent variables
set.seed(222)
control <- trainControl(method = "cv", number = 5, savePredictions = "final", 
                        allowParallel = TRUE) # Check for cross-validation
fits <- caretList(Purchase ~ ., trControl = control, methodList = 
                    c("lm", "rpart", "rf", "glm", "gbm", "knn", "svmLinear"), 
                  data = train_set, tuneList = NULL, continue_on_fail = FALSE)

# Print RMSE of each model for training data
model_results <- data.frame(
  lm = min(fits$lm$results$RMSE),
  rpart = min(fits$rpart$results$RMSE),
  rf = min(fits$rf$results$RMSE),
  glm = min(fits$glm$results$RMSE),
  gbm = min(fits$gbm$results$RMSE),
  knn = min(fits$knn$results$RMSE),
  svmLinear = min(fits$svmLinear$results$RMSE))
print(model_results)

# Resample performance of models
resamples <- resamples(fits)
dotplot(resamples, metric = "RMSE")

# Create ensemble of models by performing a linear combination
ensemble_a <- caretEnsemble(fits, metric = "RMSE", trControl = control)
summary(ensemble_a)

# Check correlation of models
modelCor(resamples)

# Create ensemble of models by using glmnet ("meta model")
ensemble_b <- caretStack(fits, 
                         method = "glmnet", 
                         metric = "RMSE", 
                         trControl = control)
print(ensemble_b)

# Validate model on test data
# Predict on test data
pred_lm <- predict.train(fits$lm, newdata = test_set)
pred_rpart <- predict.train(fits$rpart, newdata = test_set)
pred_rf <- predict.train(fits$rf, newdata = test_set)
pred_glm <- predict.train(fits$glm, newdata = test_set)
pred_gbm <- predict.train(fits$gbm, newdata = test_set)
pred_knn <- predict.train(fits$knn, newdata = test_set)
pred_svmLinear <- predict.train(fits$svmLinear, newdata = test_set)
predict_ensa <- predict(ensemble_a, newdata = test_set)
predict_ensb <- predict(ensemble_b, newdata = test_set)

# Get RMSE for test data
pred_RMSE <- data.frame(ensemble_a = RMSE(predict_ensa, test_set$Purchase),
                        ensemble_b = RMSE(predict_ensb, test_set$Purchase),                        
                        lm = RMSE(pred_lm, test_set$Purchase),
                        rf = RMSE(pred_rf, test_set$Purchase),
                        glm = RMSE(pred_glm, test_set$Purchase),
                        gbm = RMSE(pred_gbm, test_set$Purchase),
                        knn = RMSE(pred_knn, test_set$Purchase),
                        svmLinear = RMSE(pred_svmLinear, test_set$Purchase))
print(pred_RMSE)

# Perform stepwise regression to reduce independent variables 
lm_fit <- lm(Purchase ~ ., data = train_set)
step <- stepAIC(lm_fit, direction = "both")
step$anova

# Step 2: Analysis of a selected set of independent variables
# Build machine learning model for training data with reduced independent variables
set.seed(222)
control <- trainControl(method = "cv", number = 5, savePredictions = "final", 
                        allowParallel = TRUE) # Check for cross-validation
fits_2 <- caretList(Purchase ~ Gender + Age + City_Category, trControl = control, methodList = 
                      c("lm", "rpart", "rf", "glm", "gbm", "knn", "svmLinear"), 
                    data = train_set, tuneList = NULL, continue_on_fail = FALSE)

# Print RMSE of each model for training data using reduced independent variables
model_results_2 <- data.frame(
  lm = min(fits_2$lm$results$RMSE),
  rpart = min(fits_2$rpart$results$RMSE),
  rf = min(fits_2$rf$results$RMSE),
  glm = min(fits_2$glm$results$RMSE),
  gbm = min(fits_2$gbm$results$RMSE),
  knn = min(fits_2$knn$results$RMSE),
  svmLinear = min(fits_2$svmLinear$results$RMSE))
print(model_results_2)

# Resample performance of new models
resamples_2 <- resamples(fits_2)
dotplot(resamples_2, metric = "RMSE")

# Create a new ensemble of models by performing a linear combination
ensemble_2a <- caretEnsemble(fits_2, metric = "RMSE", trControl = control)
summary(ensemble_2a)

# Create ensemble of models by using glmnet ("meta model")
ensemble_2b <- caretStack(fits_2, 
                          method = "glmnet", 
                          metric = "RMSE", 
                          trControl = control)
print(ensemble_2b)

# Validate reduced model on test data
# Predict on test data
pred_lm2 <- predict.train(fits_2$lm, newdata = test_set)
pred_rpart2 <- predict.train(fits_2$rpart, newdata = test_set)
pred_rf2 <- predict.train(fits_2$rf, newdata = test_set)
pred_glm2 <- predict.train(fits_2$glm, newdata = test_set)
pred_gbm2 <- predict.train(fits_2$gbm, newdata = test_set)
pred_knn2 <- predict.train(fits_2$knn, newdata = test_set)
pred_svmLinear2 <- predict.train(fits_2$svmLinear, newdata = test_set)
predict_ens2a <- predict(ensemble_2a, newdata = test_set)
predict_ens2b <- predict(ensemble_2b, newdata = test_set)

# Get RMSE for test data
pred_RMSE_2 <- data.frame(ensemble_2a = RMSE(predict_ens2a, test_set$Purchase),
                          ensemble_2b = RMSE(predict_ens2b, test_set$Purchase),                        
                          lm_2 = RMSE(pred_lm2, test_set$Purchase),
                          rf_2 = RMSE(pred_rf2, test_set$Purchase),
                          glm_2 = RMSE(pred_glm2, test_set$Purchase),
                          gbm_2 = RMSE(pred_gbm2, test_set$Purchase),
                          knn_2 = RMSE(pred_knn2, test_set$Purchase),
                          svmLinear_2 = RMSE(pred_svmLinear2, test_set$Purchase))
print(pred_RMSE_2)
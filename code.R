#To open the csv file:
setwd(dirname(file.choose()))
getwd()
stud <- read.csv("u2563353_DS7006_CW2_data.csv", stringsAsFactors = FALSE)

head(stud)
str(stud)

library(Amelia)install.packages(c("class", "dplyr", "caret", "e1071", "randomForest", "pROC", "ggplot2", "Amelia", "corrplot"))

# Load necessary libraries
library(psych)
library(Amelia)
library(gridExtra)

# Check for missing values
apply(stud, MARGIN = 2, FUN = function(x) sum(is.na(x)))
missmap(stud, col = c("black", "grey"), legend = FALSE)

# dataset summary
summary(stud)

# Detailed statistical summary
describe(stud)


#classify the variables into categorical and numerical variables
#select the numerical variables
numeric_var <-stud %>%
  select("age","trtbps","chol","thalachh","oldpeak")
#select the categorical values
categorical_var<- stud %>%
  select("sex","cp","fbs","restecg","exng","slp","caa",
         "thall","output")%>%
  mutate_if(is.numeric, as.factor)

library(corrplot)

# Compute the correlation matrix
cor_matrix <- cor(stud[, numeric_cols])

# Visualize the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)


str(stud)
summary(stud)
head(stud)

# Histograms for continuous variables
par(mfrow = c(2, 2))
hist(stud$age, main = "Age Distribution", xlab = "Age", col = "lightblue", border = "black")
hist(stud$trtbps, main = "Blood Pressure Distribution", xlab = "Blood Pressure", col = "lightgreen", border = "black")
hist(stud$chol, main = "Cholesterol Distribution", xlab = "Cholesterol", col = "lightcoral", border = "black")
hist(stud$thalachh, main = "Max Heart Rate Distribution", xlab = "Max Heart Rate", col = "lightyellow", border = "black")

# Boxplots for continuous variables
par(mfrow = c(2, 2))
boxplot(stud$age, main = "Age Boxplot", ylab = "Age")
boxplot(stud$trtbps, main = "Blood Pressure Boxplot", ylab = "Blood Pressure")
boxplot(stud$chol, main = "Cholesterol Boxplot", ylab = "Cholesterol")
boxplot(stud$thalachh, main = "Max Heart Rate Boxplot", ylab = "Max Heart Rate")

  # Distribution of categorical variables
par(mfrow = c(2, 2))
barplot(table(stud$sex), main = "Gender Distribution", xlab = "Sex", ylab = "Count", col = "lightblue")
barplot(table(stud$cp), main = "Chest Pain Type Distribution", xlab = "Chest Pain Type", ylab = "Count", col = "lightgreen")
barplot(table(stud$fbs), main = "Fasting Blood Sugar Distribution", xlab = "Fasting Blood Sugar", ylab = "Count", col = "lightcoral")
barplot(table(stud$restecg), main = "Resting Electrocardiographic Results Distribution", xlab = "Resting ECG Results", ylab = "Count", col = "lightyellow")

# Checking for outliers
par(mfrow = c(2, 2))
plot(stud$age, stud$trtbps, main = "Age vs Blood Pressure", xlab = "Age", ylab = "Blood Pressure")
plot(stud$age, stud$chol, main = "Age vs Cholesterol", xlab = "Age", ylab = "Cholesterol")
plot(stud$age, stud$thalachh, main = "Age vs Max Heart Rate", xlab = "Age", ylab = "Max Heart Rate")
plot(stud$trtbps, stud$chol, main = "Blood Pressure vs Cholesterol", xlab = "Blood Pressure", ylab = "Cholesterol")

# Relationship with the target variable 'output'
par(mfrow = c(2, 2))
boxplot(age ~ output, data = stud, main = "Age by Heart Disease Status", xlab = "Heart Disease Status", ylab = "Age")
boxplot(trtbps ~ output, data = stud, main = "Blood Pressure by Heart Disease Status", xlab = "Heart Disease Status", ylab = "Blood Pressure")
boxplot(chol ~ output, data = stud, main = "Cholesterol by Heart Disease Status", xlab = "Heart Disease Status", ylab = "Cholesterol")
boxplot(thalachh ~ output, data = stud, main = "Max Heart Rate by Heart Disease Status", xlab = "Heart Disease Status", ylab = "Max Heart Rate")


#----------------------------------------------------------------------------------------------------------------

# Load necessary libraries
library(caret)         # For createDataPartition and confusionMatrix
library(pROC)          # For ROC and AUC
library(ggplot2)       # For plotting coefficients and feature importance
library(pheatmap)      # For heatmap
library(class)         # For k-NN
library(randomForest)  # For Random Forest
library(e1071)         # For SVM

# Set seed for reproducibility
set.seed(123)

# Split data into training and testing sets
trainIndex <- createDataPartition(stud$output, p = .8, list = FALSE, times = 1)
trainData <- stud[trainIndex, ]
testData  <- stud[-trainIndex, ]

# Define features and labels
trainX <- trainData[, -which(names(trainData) == "output")]
trainY <- trainData$output
testX <- testData[, -which(names(testData) == "output")]
testY <- testData$output

# Fit logistic regression model
logit_model <- glm(output ~ ., data = trainData, family = binomial)
summary(logit_model)

# Make predictions
prob_predictions_logit <- predict(logit_model, newdata = testData, type = "response")
predictions_logit <- ifelse(prob_predictions_logit > 0.5, 1, 0)

# Evaluate logistic regression model
conf_matrix_logistic <- confusionMatrix(as.factor(predictions_logit), as.factor(testY))
print(conf_matrix_logistic)

# Convert confusion matrix to matrix format for pheatmap
conf_matrix_matrix_logistic <- as.matrix(conf_matrix_logistic$table)

# Define the function to plot confusion matrix heatmap
plot_cm_heatmap <- function(conf_matrix, title) {
  pheatmap(conf_matrix, display_numbers = TRUE, main = title, 
           color = colorRampPalette(c("white", "blue"))(50))
}

# Plot heatmaps for logistic regression
plot_cm_heatmap(conf_matrix_matrix_logistic, "Confusion Matrix - Logistic Regression")

# Calculate and plot ROC curve
roc_curve_logit <- roc(testY, prob_predictions_logit)
auc_value_logistic <- auc(roc_curve_logit)
print(paste("AUC (Logistic Regression):", round(auc_value_logistic, 2)))
plot(roc_curve_logit, main = paste("ROC Curve (Logistic Regression, AUC = ", round(auc_value_logistic, 2), ")", sep=""))

# Plot feature coefficients for logistic regression
coefficients_logit <- summary(logit_model)$coefficients
coef_df <- data.frame(Feature = rownames(coefficients_logit), Coefficient = coefficients_logit[, "Estimate"])
ggplot(coef_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients", x = "Feature", y = "Coefficient")

# Train k-NN model
k <- 5
knn_predictions <- knn(train = trainX, test = testX, cl = trainY, k = k)

# Evaluate k-NN model
conf_matrix_knn <- confusionMatrix(as.factor(knn_predictions), as.factor(testY))
print(conf_matrix_knn)

# Note: k-NN does not provide probability predictions, so we skip ROC/AUC for k-NN.

# Hyperparameter tuning for k-NN
tune_knn <- function(k_values) {
  results <- data.frame(k = integer(), Accuracy = numeric())
  
  for (k in k_values) {
    knn_pred <- knn(train = trainX, test = testX, cl = trainY, k = k)
    conf_matrix <- confusionMatrix(as.factor(knn_pred), as.factor(testY))
    accuracy <- conf_matrix$overall['Accuracy']
    results <- rbind(results, data.frame(k = k, Accuracy = accuracy))
  }
  
  return(results)
}

k_values <- 1:10
tuning_results <- tune_knn(k_values)
print(tuning_results)

# Plot k-NN accuracy vs. k
ggplot(tuning_results, aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "k-NN Accuracy vs. k", x = "k", y = "Accuracy")

# Train SVM model
svm_model <- svm(trainX, as.factor(trainY), kernel = "radial", cost = 1, gamma = 0.1)
summary(svm_model)

# Make predictions and evaluate SVM
svm_predictions <- predict(svm_model, testX)
conf_matrix_svm <- confusionMatrix(as.factor(svm_predictions), as.factor(testY))
print(conf_matrix_svm)

# Make predictions and compute ROC for SVM
decision_values <- attr(predict(svm_model, testX, decision.values = TRUE), "decision.values")
roc_curve_svm <- roc(testY, decision_values)
auc_value_svm <- auc(roc_curve_svm)
print(paste("AUC (SVM):", round(auc_value_svm, 2)))
plot(roc_curve_svm, main = paste("ROC Curve (SVM, AUC = ", round(auc_value_svm, 2), ")", sep=""))

# Plot heatmaps for SVM
plot_cm_heatmap(as.matrix(conf_matrix_svm$table), "Confusion Matrix - SVM")

# Train Random Forest model
rf_model <- randomForest(x = trainX, y = as.factor(trainY), ntree = 100, importance = TRUE)
print(rf_model)

# Make predictions and evaluate Random Forest
rf_predictions <- predict(rf_model, testX)
conf_matrix_rf <- confusionMatrix(as.factor(rf_predictions), as.factor(testY))
print(conf_matrix_rf)

# Calculate and plot ROC curve for Random Forest
prob_predictions_rf <- predict(rf_model, testX, type = "prob")[, 2]
roc_curve_rf <- roc(testY, prob_predictions_rf)
auc_value_rf <- auc(roc_curve_rf)
print(paste("AUC (Random Forest):", round(auc_value_rf, 2)))
plot(roc_curve_rf, main = paste("ROC Curve (Random Forest, AUC = ", round(auc_value_rf, 2), ")", sep=""))

# Plot feature importance for Random Forest
importance_df_rf <- as.data.frame(importance(rf_model))
ggplot(importance_df_rf, aes(x = reorder(rownames(importance_df_rf), MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Feature", y = "Importance (MeanDecreaseGini)")

# Plot heatmaps for Random Forest
plot_cm_heatmap(as.matrix(conf_matrix_rf$table), "Confusion Matrix - Random Forest")

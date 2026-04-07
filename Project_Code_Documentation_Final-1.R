library(tidyverse)
library(rvest)
library(caret)
library(GGally)
library(glmnet)
library(pROC)
library(xgboost)
library(iml)
library(corrplot)
library(reshape2)

# ===========================
# DATA IMPORT AND BASIC CLEANING
# ===========================

wisconsin <- read.csv("~/Documents/STAT 4630/bcwisconsin/wdbc.data", header = FALSE)

#1 mean, 2 standard error, 3 worst

names <- c("ID",
           "Diagnosis",
           "radius_mean",
           "texture_mean",
           "perimeter_mean",
           "area_mean",
           "smoothness_mean",
           "compactness_mean",
           "concavity_mean",
           "concave_points_mean",
           "symmetry_mean",
           "fractal_dimension_mean",
           "radius_se",
           "texture_se",
           "perimeter_se",
           "area_se",
           "smoothness_se",
           "compactness_se",
           "concavity_se",
           "concave_points_se",
           "symmetry_se",
           "fractal_dimension_se",
           "radius_worst",
           "texture_worst",
           "perimeter_worst",
           "area_worst",
           "smoothness_worst",
           "compactness_worst",
           "concavity_worst",
           "concave_points_worst",
           "symmetry_worst",
           "fractal_dimension_worst")

colnames(wisconsin) <- names

wisconsin$Diagnosis <- factor(wisconsin$Diagnosis, levels = c("M", "B"))

wisconsin <- subset(wisconsin, select = -c(ID))

# ===========================
# TRAIN/TEST SPLIT W STRATIFICATION
# ===========================

set.seed(4630)

# 80/20 split
train_index <- createDataPartition(wisconsin$Diagnosis,
                                   p = 0.8,  
                                   list = FALSE)

traindata <- wisconsin[train_index, ]
testdata  <- wisconsin[-train_index, ]

# Verifying class proportions are the same
prop.table(table(wisconsin$Diagnosis))
prop.table(table(traindata$Diagnosis))
prop.table(table(testdata$Diagnosis))

# ===========================
# STRATIFIED CV SETUP
# =========================== 

cv_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  # Mitigating class imbalance further
  sampling = "up"  
)

# ===========================
# NORMALIZING PREDICTORS
# ===========================

# Build preprocessing recipe
preprocess_steps <- preProcess(
  traindata %>% select(-Diagnosis),
  method = c("center", "scale")
)

# Apply to train and test
trainscaled <- traindata
trainscaled[, -1] <- predict(preprocess_steps, traindata[, -1])

testscaled <- testdata
testscaled[, -1] <- predict(preprocess_steps, testdata[, -1])

# ===========================
# CORRELATION MATRIX
# =========================== 

numeric_predictors <- trainscaled %>% select(-Diagnosis)

cormat <- round(cor(numeric_predictors), 2)

# Melt for ggplot
melted_cormat <- melt(cormat)

# Create heatmap
ggplot(melted_cormat, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "darkorange3",
    midpoint = 0,
    limit = c(-1, 1),
    name = "Correlation"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    axis.text.y = element_text(size = 7),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid = element_blank()
  ) +
  coord_fixed() +
  labs(
    title = "Correlation Heatmap of Standardized Predictors",
    x = "",
    y = ""
  )



# ===========================
# EDA
# =========================== 

# Basic summaries
summary(traindata)

# Class distribution plot
traindata %>%
  ggplot(aes(x = Diagnosis, fill = Diagnosis)) +
  geom_bar() +
  labs(title = "Class Distribution in Training Set") +
  theme_minimal()

# Pairwise correlations (top 10 correlated with radius_mean)
top_corr_vars <- numeric_predictors %>%
  cor() %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  select(feature, radius_mean = radius_mean) %>%
  arrange(desc(abs(radius_mean))) %>%
  slice(1:10) %>%
  pull(feature)

pairs_data <- trainscaled[, c("Diagnosis", top_corr_vars)]

GGally::ggpairs(pairs_data, aes(color = Diagnosis, alpha = 0.4))

# PCA visualization
pca <- prcomp(numeric_predictors, center = TRUE, scale = TRUE)
pca_df <- data.frame(pca$x, Diagnosis = trainscaled$Diagnosis)

pca_df %>%
  ggplot(aes(x = PC1, y = PC2, color = Diagnosis)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA: PC1 vs PC2") +
  theme_minimal()

# ===========================
# KNN 
# ===========================

# Model
set.seed(4630)

knn_model <- train(
  Diagnosis ~ .,
  data = trainscaled,
  method = "knn",
  metric = "ROC",             
  trControl = cv_ctrl,
  tuneLength = 20             
)

knn_model
plot(knn_model, xlab = "# Neighbors")

# Evaluate on test set

knn_preds <- predict(knn_model, newdata = testscaled)
knn_probs <- predict(knn_model, newdata = testscaled, type = "prob")

confusionMatrix(knn_preds, testscaled$Diagnosis)

# ROC/AUC
pROC::roc(response = testscaled$Diagnosis,
          predictor = knn_probs$M,
          levels = rev(levels(testscaled$Diagnosis))) %>%
  pROC::auc()


# ===========================
# LOGISTIC REGRESSION
# ===========================

# Creating train/test for x and y
x.train <- model.matrix(Diagnosis ~ ., trainscaled)[,-1]
y.train <- ifelse(trainscaled$Diagnosis == "M", 1, 0)

x.test <- model.matrix(Diagnosis ~ ., testscaled)[,-1]
y.test <- ifelse(testscaled$Diagnosis == "M", 1, 0)

# Fit logistic regression on the training set
glm.fit <- glm(Diagnosis ~ ., data = trainscaled, family = "binomial")

# Extracting coefficients
glm.coef <- summary(glm.fit)$coef
odds.ratios <- exp(glm.coef[,1])

# Predicted probabilities 
glm.probs <- predict(glm.fit, testscaled, type = "response")

# Convert predicted probabilities to class labels (threshold = 0.5)
glm.pred <- rep("B", nrow(testscaled))
glm.pred[glm.probs > 0.5] <- "M"
glm.pred <- factor(glm.pred, levels = c("M", "B"))

# Calculating misclassification rate
glm.error <- mean(glm.pred != testscaled$Diagnosis)

# Creating confusion matrix
glm.cm <- confusionMatrix(glm.pred, testscaled$Diagnosis, positive = "M")
glm.cm

# Calculating AUC/ROC
roc.glm <- roc(y.test, as.numeric(glm.probs))
auc.glm <- auc(roc.glm)

auc.glm

# Printing important metrics
cat("\n===========================\nLOGISTIC REGRESSION RESULTS\n===========================\n")
cat("Coefficients Table:\n")
print(glm.coef)
cat("\nOdds Ratios (exp(beta)):\n")
print(odds.ratios)
cat("\nClassification Error:", glm.error, "\n")
cat("Confusion Matrix:\n")
print(glm.cm)
plot(roc.glm, main="ROC Curve")
cat("AUC:\n", auc.glm, "\n")

# ===========================
# LASSO
# ===========================

# Fit lasso on the training set and examine the coefficient path
grid <- 10^seq(3, -3, length = 100)
lasso.mod <- glmnet(x.train, y.train, alpha = 1, lambda = grid, family="binomial")
plot(lasso.mod)

# 10 fold CV to choose lambda (lambda.min)
set.seed(1)
cv.lasso <- cv.glmnet(x.train, y.train, alpha = 1, family = "binomial", type.measure = "class")
plot(cv.lasso)
bestlam.lasso <- cv.lasso$lambda.min

# Finding classification error and the number of nonzero coefficients at the selected lambda
lasso.prob <- predict(lasso.mod, s = bestlam.lasso, newx = x.test, type = "response")
lasso.class <- ifelse(lasso.prob > 0.5, 1, 0)

lasso.error <- mean(lasso.class != y.test)

lasso.coef <- predict(lasso.mod, type = "coefficients", s = bestlam.lasso)[,1]

nonzero.coef.count <- length(which(lasso.coef != 0)) - 1
nonzero.coef.vals <- lasso.coef[lasso.coef != 0]

# Calculating AUC/ROC
roc.lasso <- roc(y.test, as.numeric(lasso.prob))
auc.lasso <- auc(roc.lasso)

# Creating confusion matrix
lasso.class.bm <- factor(ifelse(lasso.class == 1, "M", "B"), levels = c("M", "B")) # Converting labels back to "M" and "B"
y.test.bm <- factor(ifelse(y.test == 1, "M", "B"), levels = c("M", "B")) # Converting labels back to "M" and "B"
lasso.cm = confusionMatrix(lasso.class.bm, y.test.bm, positive = "M")

# Printing important metrics
cat("\n=============\nLASSO RESULTS\n=============\n")
cat("Lambda.min:", bestlam.lasso, "\n")
cat("Classification Error:", lasso.error, "\n")
cat("Number of Nonzero Coefficients (excl. intercept):", nonzero.coef.count, "\n")
cat("Nonzero Coefficient Values (incl. intercept):\n")
print(nonzero.coef.vals)
cat("Confusion Matrix:\n")
print(lasso.cm)
plot(roc.lasso, main="ROC Curve")
cat("AUC:", auc.lasso)

# ===========================
# RANDOM FOREST 
# ===========================

set.seed(4630)

# Tuning grid: mtry = number of variables considered at each split
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10, 12, 15)
)

rf_model <- train(
  Diagnosis ~ .,
  data = trainscaled,
  method = "rf",
  metric = "ROC",
  trControl = cv_ctrl,
  tuneGrid = rf_grid,
  ntree = 500,
  importance = TRUE
)

# View tuning results
rf_model
plot(rf_model, xlab = "mtry")

# Predictions
rf_preds <- predict(rf_model, newdata = testscaled)
rf_probs <- predict(rf_model, newdata = testscaled, type = "prob")

# Confusion matrix
confusionMatrix(rf_preds, testscaled$Diagnosis)

# ROC / AUC
rf_auc <- pROC::roc(
  response = testscaled$Diagnosis,
  predictor = rf_probs$M,
  levels = rev(levels(testscaled$Diagnosis))
) %>% pROC::auc()

rf_auc

# ===========================
# BOOSTING (XGBoost) + SHAP 
# ===========================

library(xgboost)
library(pROC)
library(tidyverse)

set.seed(4630)

# Convert response to numeric for xgboost
y_train_xgb <- ifelse(trainscaled$Diagnosis == "M", 1, 0)
y_test_xgb  <- ifelse(testscaled$Diagnosis == "M", 1, 0)

x_train_xgb <- as.matrix(trainscaled %>% select(-Diagnosis))
x_test_xgb  <- as.matrix(testscaled %>% select(-Diagnosis))

# ===========================
# INTERNAL VALIDATION SPLIT
# ===========================
val_index <- sample(1:nrow(x_train_xgb), size = floor(0.15 * nrow(x_train_xgb)))

x_val_xgb <- x_train_xgb[val_index, ]
y_val_xgb <- y_train_xgb[val_index]

x_train_sub <- x_train_xgb[-val_index, ]
y_train_sub <- y_train_xgb[-val_index]

dtrain <- xgb.DMatrix(data = x_train_sub, label = y_train_sub)
dval   <- xgb.DMatrix(data = x_val_xgb, label = y_val_xgb)
dtest  <- xgb.DMatrix(data = x_test_xgb, label = y_test_xgb)

# ===========================
# HYPERPARAMETER GRID
# ===========================
xgb_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 4, 5),
  nrounds = c(200, 400),
  gamma = c(0, 1),
  min_child_weight = c(1, 3)
)

best_auc <- 0
best_model <- NULL
best_params <- NULL

# ===========================
# TUNING LOOP
# ===========================
for (i in 1:nrow(xgb_grid)) {
  
  params <- list(
    booster = "gbtree",
    eta = xgb_grid$eta[i],
    max_depth = xgb_grid$max_depth[i],
    gamma = xgb_grid$gamma[i],
    min_child_weight = xgb_grid$min_child_weight[i],
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
  
  xgb_fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = xgb_grid$nrounds[i],
    evals = list(val = dval),
    verbose = 0
  )
  
  val_preds <- predict(xgb_fit, dval)
  auc_val <- auc(roc(y_val_xgb, val_preds))
  
  if (auc_val > best_auc) {
    best_auc <- auc_val
    best_model <- xgb_fit
    best_params <- params
  }
}

cat("\nBest validation AUC:", best_auc, "\n\n")

# ===========================
# TEST SET PERFORMANCE
# ===========================

xgb_probs <- predict(best_model, dtest)
xgb_class <- factor(ifelse(xgb_probs > 0.5, "M", "B"), levels = c("M","B"))

cat("\nConfusion Matrix:\n")
print(confusionMatrix(xgb_class, testscaled$Diagnosis, positive = "M"))

xgb_roc <- roc(response = y_test_xgb, predictor = xgb_probs)
cat("\nTest AUC:", auc(xgb_roc), "\n")

# ===========================
# SHAP VALUES (XGBoost built-in)
# ===========================

# shapcontrib returns matrix: rows = obs, columns = features + bias
shap_values <- predict(best_model, x_train_xgb, predcontrib = TRUE)

# remove the last column (BIAS term)
shap_values <- shap_values[, -ncol(shap_values)]

feature_names <- colnames(x_train_xgb)

# Compute SHAP importance (mean |SHAP|)
shap_importance <- data.frame(
  Feature = feature_names,
  MeanAbsSHAP = apply(abs(shap_values), 2, mean)
) %>%
  arrange(desc(MeanAbsSHAP))

cat("\nSHAP Feature Importance:\n")
print(shap_importance)

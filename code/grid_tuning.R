### imbalanced sample
library(ROSE)
cat("The proportion of 1 in train data set is :", prop.table(table(car_train$loan_default))[2])

# new_data <- ovun.sample(formula = loan_default ~ ., data = car_train, method = "over",
#                         N = 1.2 * nrow(car_train), seed = 2333)$data %>% as_tibble()
# new_data <- ROSE(formula = loan_default ~ ., data = car_train, 
#                  N = 1.2 * nrow(car_train), seed = 2333)

# define a eval function
lgb_f1_macro <- function(preds, dtrain) {
  
  f1_macro <- f1_score(y_pred = ifelse(preds > 0.25, 1, 0), 
                       y_true = dtrain$getinfo("label"), 
                       pattern = "macro")$f1_macro
  
  res <- list(name = "f1_macro", 
              value = ifelse(is.nan(f1_macro), 0, f1_macro),
              higher_better = TRUE)
  return(res)
}

### tuning parameter
# use all data in train dataset
x.train <- car_train %>% select(-loan_default)
y.train <- car_train[["loan_default"]]

x.train <- x_train
y.train <- y_train

feas <- colnames(x.train)
cate_feas <- cate_feature


## Stage 1: tune complexity of tree
# 1. max_depth, num_leaves
param_grid <- expand_grid(max_depth = c(6, 7, 8), num_leaves = c(2^5, 2^6, 2^7)) 
metric_df_1 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = param_grid$max_depth[i], 
                              num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_1 <- rbind(metric_df_1, metric)
  colnames(metric_df_1) <- c("valid_eval", "valid_err")
  
}

cat("best max_depth:", param_grid$max_depth[which.max(metric_df_1$valid_eval)], "\n",
    "best num_leaves:", param_grid$num_leaves[which.max(metric_df_1$valid_eval)], "\n")



# 2. details of num_leaves
param_grid <- expand_grid(num_leaves = seq(100, 150, 10)) 
metric_df_2 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 7, num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_2 <- rbind(metric_df_2, metric)
  colnames(metric_df_2) <- c("valid_eval", "valid_err")
  
}

cat("When max_depth =", 7, "best num_leaves is:", param_grid$num_leaves[which.max(metric_df_2$valid_eval)], "\n")


## 3. min_child_weight, min_data_in_leaf
param_grid <- expand_grid(min_child_weight = seq(1, 10, 2)) 
metric_df_3 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 7, num_leaves = 100, 
                              min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_3 <- rbind(metric_df_3, metric)
  colnames(metric_df_3) <- c("valid_eval", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_3$valid_eval)], "\n")



## 4. details of min_child_weight
param_grid <- expand_grid(min_child_weight = seq(7, 10, 1)) 
metric_df_4 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 7, num_leaves = 100, 
                              min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_4 <- rbind(metric_df_4, metric)
  colnames(metric_df_4) <- c("valid_eval", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_4$valid_eval)], "\n")




## Stage 2: tune sampling parameters
param_grid <- expand_grid(bagging_fraction = c(0.7, 0.8, 0.9), feature_fraction = c(0.7, 0.8)) 
metric_df_5 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = param_grid$feature_fraction[i],
                              bagging_fraction = param_grid$bagging_fraction[i], 
                              bagging_freq = 5, max_depth = 7, num_leaves = 100, 
                              min_child_weight = 8),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_5 <- rbind(metric_df_5, metric)
  colnames(metric_df_5) <- c("valid_eval", "valid_err")
  
}


cat("best bagging_fraction:", param_grid$bagging_fraction[which.max(metric_df_5$valid_eval)], "\n", 
    "best feature_fraction:", param_grid$feature_fraction[which.max(metric_df_5$valid_eval)])





## Stage 3: tune learning rate and number of iterations
# learning rate, nrounds
param_grid <- expand_grid(learning_rate = c(0.005, 0.01, 0.05), nrounds = c(1000, 2000, 3000)) 
metric_df_6 <- data.frame()

for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", 
                              learning_rate = param_grid$learning_rate[i], 
                              feature_fraction = 0.7, bagging_fraction = 0.9, 
                              bagging_freq = 5, max_depth = 7, num_leaves = 100, 
                              min_child_weight = 8),
                data = dtrain, nrounds = param_grid$nrounds[i], 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                eval = lgb_f1_macro, categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(max(as.numeric(clf$record_evals$valid$f1_macro$eval)),
              clf$record_evals$valid$f1_macro$eval_err[[which.max(as.numeric(clf$record_evals$valid$f1_macro$eval))]])
  
  metric_df_6 <- rbind(metric_df_6, metric)
  colnames(metric_df_6) <- c("valid_eval", "valid_err")
  
}

cat("best learning_rate:", param_grid$learning_rate[which.max(metric_df_6$valid_eval)], "\n", 
    "best nrounds:", param_grid$nrounds[which.max(metric_df_6$valid_eval)])





### the best combination of parameter 
dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
param <- list(objective = "binary", boosting = "gbdt", 
              learning_rate = 0.01, 
              feature_fraction = 0.7, bagging_fraction = 0.9, 
              bagging_freq = 5, max_depth = 7, num_leaves = 100, 
              min_child_weight = 8)

opt.clf.2 <- lgb.train(data = dtrain, params = param, 
                     nrounds = 1000, categorical_feature = cate_feas, 
                     verbose = -1, force_row_wise = TRUE)


pred_valid <- predict(opt.clf.2,  data = as.matrix(x_valid[feas]))
Meas_valid <- f1_score(y_pred = ifelse(pred_valid > 0.244, 1, 0), 
                       y_true = y_valid, pattern = "macro") # 0.5862859



# save submit
pred_test <- predict(opt.clf.2, data = as.matrix(car_test[feas]))
submit <- data.frame(customer_id = test$customer_id, loan_default = ifelse(pred_test > 0.244, 1, 0))
write.csv(submit, file = "./res/res18_opt.csv", row.names = FALSE)


###### 
# 1. use all features & train prop = 0.8 (seed = 233)
# 1) eval = "auc", thr = 0.2495, f1_score_test = 0.58054 
# list(objective = "binary", boosting = "gbdt", 
#      learning_rate = 0.005, 
#      feature_fraction = 0.7, bagging_fraction = 0.8, 
#      bagging_freq = 5, max_depth = 7, num_leaves = 110, 
#      min_child_weight = 7) ; nrounds = 3000

# 2) eval  = lgb_f1_score, thr 
# list(objective = "binary", boosting = "gbdt", 
#      learning_rate = 0.01, 
#      feature_fraction = 0.7, bagging_fraction = 0.9, 
#      bagging_freq = 5, max_depth = 7, num_leaves = 100, 
#      min_child_weight = 8) ; nrounds = 1000

### imbalanced sample
library(ROSE)
cat("The proportion of 1 in train data set is :", prop.table(table(car_train$loan_default))[2])

# new_data <- ovun.sample(formula = loan_default ~ ., data = car_train, method = "over",
#                         N = 1.2 * nrow(car_train), seed = 2333)$data %>% as_tibble()
# new_data <- ROSE(formula = loan_default ~ ., data = car_train, 
#                  N = 1.2 * nrow(car_train), seed = 2333)

### tuning parameter
feas <- freq_feas_0
cate_feas <- freq_cat_feas_0

# use all data in train dataset
x.train <- car_train %>% select(-loan_default, -customer_id)
y.train <- car_train[["loan_default"]]

## Stage 1: tune complexity of tree
# 1. max_depth, num_leaves
param_grid <- expand_grid(max_depth = c(6, 7, 8), num_leaves = c(50, 100, 200)) 

metric_df_1 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = param_grid$max_depth[i], 
                              num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_1 <- rbind(metric_df_1, metric)
  colnames(metric_df_1) <- c("valid_auc", "valid_err")
  
}

cat("best max_depth:", param_grid$max_depth[which.max(metric_df_1$valid_auc)], "\n",
    "best num_leaves:", param_grid$num_leaves[which.max(metric_df_1$valid_auc)], "\n")



# 2. details of num_leaves
param_grid <- expand_grid(num_leaves = seq(160, 240, 20)) 
metric_df_2 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 8, num_leaves = param_grid$num_leaves[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_2 <- rbind(metric_df_2, metric)
  colnames(metric_df_2) <- c("valid_auc", "valid_err")
  
}

cat("best max_depth:", 8, "best num_leaves:", param_grid$num_leaves[which.max(metric_df_2$valid_auc)], "\n")


## 3. min_child_weight, min_data_in_leaf
param_grid <- expand_grid(min_child_weight = seq(1, 10, 3)) 
metric_df_3 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 8, num_leaves = 240, 
                              min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_3 <- rbind(metric_df_3, metric)
  colnames(metric_df_3) <- c("valid_auc", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_3$valid_auc)], "\n")



## 4. details of min_child_weight
param_grid <- expand_grid(min_child_weight = seq(1, 4, 1)) 
metric_df_4 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5, 
                              max_depth = 8, num_leaves = 240, 
                              min_child_weight = param_grid$min_child_weight[i]),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_4 <- rbind(metric_df_4, metric)
  colnames(metric_df_4) <- c("valid_auc", "valid_err")
  
}

cat("best min_child_weight:", param_grid$min_child_weight[which.max(metric_df_4$valid_auc)], "\n")




## Stage 2: tune sampling parameters
param_grid <- expand_grid(bagging_fraction = c(0.7, 0.8, 0.9), feature_fraction = c(0.7, 0.8)) 
metric_df_5 <- data.frame()
for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", learning_rate = 0.1, 
                              feature_fraction = param_grid$feature_fraction[i],
                              bagging_fraction = param_grid$bagging_fraction[i], 
                              bagging_freq = 5, max_depth = 8, num_leaves = 240, 
                              min_child_weight = 1),
                data = dtrain, nrounds = 1000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_5 <- rbind(metric_df_5, metric)
  colnames(metric_df_5) <- c("valid_auc", "valid_err")
  
}


cat("best bagging_fraction:", param_grid$bagging_fraction[which.max(metric_df_5$valid_auc)], "\n", 
    "best feature_fraction:", param_grid$feature_fraction[which.max(metric_df_5$valid_auc)])





## Stage 3: tune learning rate and number of iterations
# learning rate
param_grid <- expand_grid(learning_rate = c(0.005, 0.01, 0.05)) 
metric_df_6 <- data.frame()

for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", 
                              learning_rate = param_grid$learning_rate[i], 
                              feature_fraction = 0.8, bagging_fraction = 0.9, 
                              bagging_freq = 5, max_depth = 8, num_leaves = 240, 
                              min_child_weight = 1),
                data = dtrain, nrounds = 3000, 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_6 <- rbind(metric_df_6, metric)
  colnames(metric_df_6) <- c("valid_auc", "valid_err")
  
}

cat("best learning_rate:", param_grid$learning_rate[which.max(metric_df_6$valid_auc)], "\n")



# nrounds
param_grid <- expand_grid(nrounds = seq(2000, 4000, 500)) 
metric_df_7 <- data.frame()

for (i in 1:nrow(param_grid)) {
  
  dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
  
  clf <- lgb.cv(params = list(objective = "binary", boosting = "gbdt", 
                              learning_rate = 0.05, 
                              feature_fraction = 0.8, bagging_fraction = 0.9, 
                              bagging_freq = 5, max_depth = 8, num_leaves = 240, 
                              min_child_weight = 1),
                data = dtrain, nrounds = param_grid$nrounds[i], 
                nfold = 5, stratified = TRUE, early_stopping_rounds = 50, 
                metric = "auc", categorical_feature = cate_feas, 
                verbose = -1, force_row_wise = TRUE)
  
  metric <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
  
  metric_df_7 <- rbind(metric_df_7, metric)
  colnames(metric_df_7) <- c("valid_auc", "valid_err")
  
}

cat("best nrounds:", param_grid$nrounds[which.max(metric_df_7$valid_auc)], "\n")



### the best combination of parameter for freq_feas_90
dtrain <- lgb.Dataset(as.matrix(x.train[feas]), label = y.train, free_raw_data = FALSE)
param <- list(objective = "binary", boosting = "gbdt", 
              learning_rate = 0.05, 
              feature_fraction = 0.8, bagging_fraction = 0.8, 
              bagging_freq = 5, max_depth = 8, num_leaves = 150, 
              min_child_weight = 1)
opt.clf <- lgb.train(data = dtrain, params = param, 
                     nrounds = 3000, categorical_feature = cate_feas, 
                     verbose = -1, force_row_wise = TRUE)

pred_test <- predict(opt.clf,  data = as.matrix(car_test[feas]))
submit <- data.frame(customer_id = car_test$customer_id, loan_default = ifelse(pred_test > 0.5, 1, 0))
write.csv(submit, file = "E:/Competition/xf/res/res8_opt.csv", row.names = FALSE)



### the best combination of parameter for freq_feas_90
dtrain <- lgb.Dataset(as.matrix(x.train[freq_feas_40]), label = y.train, free_raw_data = FALSE)
param <- list(objective = "binary", boosting = "gbdt", 
              learning_rate = 0.01, 
              feature_fraction = 0.7, bagging_fraction = 0.8, 
              bagging_freq = 5, max_depth = 7, num_leaves = 40, 
              min_child_weight = 5)
opt.clf <- lgb.train(data = dtrain, params = param, 
                     nrounds = 1500, categorical_feature = freq_cat_feas_40, 
                     verbose = -1, force_row_wise = TRUE)

pred_test <- predict(opt.clf,  data = as.matrix(car_test[freq_feas_40]))
submit <- data.frame(customer_id = car_test$customer_id, loan_default = ifelse(pred_test > 0.25, 1, 0))
write.csv(submit, file = "E:/Competition/xf/res/res4_opt.csv", row.names = FALSE)


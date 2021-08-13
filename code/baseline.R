
### define the performance metrics f1 score (macro and micro)
f1_score <- function(y_pred, y_true, pattern = c("micro", "macro")) {
  
  level_of_label <- length(unique(y_true))
  ConfusionMatrix <- confusionMatrix(data = factor(y_pred), reference = factor(y_true))$table
    
  if (pattern == "macro") {
    
    precision_macro <- sapply(1:level_of_label, function(l) {
      tmp_precision <- ConfusionMatrix[l, l] / (0.001 + rowSums(ConfusionMatrix)[l])})
    
    recall_macro <- sapply(1:level_of_label, function(l) {
      tmp_recall <- ConfusionMatrix[l, l] / colSums(ConfusionMatrix)[l]})
    
    f1_macro <- unique(2 * mean((precision_macro * recall_macro) / (precision_macro + recall_macro)))
    Meas <- list(df = data.frame(precision = precision_macro, recall = recall_macro), 
                 f1_macro = f1_macro)
  } 
  
  if (pattern == "micro") {
    
    precision_micro <- sum(diag(ConfusionMatrix)) / length(y_true)
    recall_micro <- sum(diag(ConfusionMatrix)) / length(y_true)
    
    f1_micro <- unique(2 * (precision_micro * recall_micro) / (precision_micro + recall_micro))
    
    Meas <- list(df = data.frame(precision = precision_micro, recall = recall_micro), 
                 f1_micro = f1_micro)
  }
  
  return(Meas)
  
}




#######################################################################################################################
##################################################### Baseline Model ##################################################
#######################################################################################################################

## sample splitting 
set.seed(233)
trainidx <- createDataPartition(car_train$loan_default, p = 0.8, list = FALSE)
x_train <- car_train[trainidx, ] %>% select(-loan_default)
y_train <- car_train[["loan_default"]][trainidx]

x_valid <- car_train[-trainidx, ] %>% select(-loan_default)
y_valid <- car_train[["loan_default"]][-trainidx]


#  parameter set
xgb_param <- list(booster = "gbtree", eta = 0.05, max_depth = 8, min_child_weight = 1.75, 
                  colsample_bytree = 0.7, subsample = 0.85)
lgb_param <- list(objective = "binary", boosting = "gbdt", learning_rate = 0.01, 
                  min_child_weight = 1, feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5)


# feature selection result
feas <- freq_feas_30
cate_feas <- freq_cat_feas_30

#######################################################################################################################
#################################### Baseline 1: xgb + classification threshold #######################################
#######################################################################################################################
dtrain <- xgb.DMatrix(as.matrix(x_train), label = y_train)
dvalid <- xgb.DMatrix(as.matrix(x_valid))

xgb_1 <- xgb.train(data = dtrain, params = param, nrounds = 1000, 
                  nthreads = 4, objective = "binary:logistic")

pred_valid <- predict(xgb_1, newdata = dvalid)
Meas_b1 <- f1_score(y_pred = ifelse(pred_valid > 0.25, 1, 0), y_true = y_valid, pattern = "macro")




#######################################################################################################################
################## Baseline 2: xgb + feature selection (NULL importance) + classification threshold ###################
#######################################################################################################################
dtrain <- xgb.DMatrix(as.matrix(x_train[feas]), label = y_train)
dvalid <- xgb.DMatrix(as.matrix(x_valid[feas]))

xgb_2 <- xgb.train(data = dtrain, params = xgb_param, nrounds = 1000, 
                   nthreads = 4, objective = "binary:logistic")

pred_valid <- predict(xgb_2, newdata = dvalid)
Meas_b2 <- f1_score(y_pred = ifelse(pred_valid > 0.25, 1, 0), y_true = y_valid, pattern = "macro")




#######################################################################################################################
############################ Baseline 3: weighting + xgb + classification threshold  ##################################
#######################################################################################################################
W1 <- 1/prop.table(table(y_train))
weight_tr <- ifelse(y_train == 0, W1[1], W1[2])
weight_va <- ifelse(y_valid == 0, W1[1], W1[2])

dtrain <- xgb.DMatrix(as.matrix(x_train), label = y_train, weight = weight_tr)
dvalid <- xgb.DMatrix(as.matrix(x_valid), weight = weight_va)

xgb_3 <- xgb.train(data = dtrain, params = xgb_param, nrounds = 1000, 
                   nthreads = 4, objective = "binary:logistic")

pred_valid <- predict(xgb_3, newdata = dvalid)
Meas_b3 <- f1_score(y_pred = ifelse(pred_valid > 0.5, 1, 0), y_true = y_valid, pattern = "macro")




#######################################################################################################################
######## Baseline 4: weighting + xgb + classification threshold + feature selection (NULL importance) #################
#######################################################################################################################
dtrain <- xgb.DMatrix(as.matrix(x_train[feas]), label = y_train, weight = weight_tr)
dvalid <- xgb.DMatrix(as.matrix(x_valid[feas]), weight = weight_va)

xgb_4 <- xgb.train(data = dtrain, params = xgb_param, nrounds = 1000, 
                   nthreads = 4, objective = "binary:logistic")

pred_valid <- predict(xgb_4, newdata = dvalid)
Meas_b4 <- f1_score(y_pred = ifelse(pred_valid > 0.5, 1, 0), y_true = y_valid, pattern = "macro")








######################################################################################################################
################################## Baseline 5: lgb + classification threshold ########################################
######################################################################################################################
dtrain <- lgb.Dataset(as.matrix(x_train), label = y_train, free_raw_data = FALSE)

cate_feature <- colnames(x_train %>% select(contains(c("_id", "_flag", "_type")), credit_level))
lgb_1 <- lgb.train(params = lgb_param, data = dtrain, nrounds = 1000, categorical_feature = cate_feature)

pred_valid <- predict(lgb_1, data = as.matrix(x_valid))
Meas_b5 <- f1_score(y_pred = ifelse(pred_valid > 0.25, 1, 0), y_true = y_valid, pattern = "macro")
# mcw = 4, nrounds = 1000, 0.5833
# mcw = 4, nrounds = 2000, 0.5827
# mcw = 3, nrounds = 1000, 0.5839
# mcw = 3, nrounds = 2000, 0.5826




######################################################################################################################
########## Baseline 6: lgb + feature selection (NULL importance) + lassification threshold ###########################
######################################################################################################################
dtrain <- lgb.Dataset(as.matrix(x_train[feas]), label = y_train, free_raw_data = FALSE)

cate_feature <- intersect(cate_feature, colnames(x_train[feas]))
lgb_2 <- lgb.train(params = lgb_param, data = dtrain, nrounds = 1000, categorical_feature = cate_feature)

pred_valid <- predict(lgb_2, data = as.matrix(x_valid[feas]))
Meas_b6 <- f1_score(y_pred = ifelse(pred_valid > 0.245, 1, 0), y_true = y_valid, pattern = "macro")




######################################################################################################################
############################## Baseline 7: lgb + weighting + lassification threshold #################################
######################################################################################################################
dtrain <- lgb.Dataset(as.matrix(x_train), label = y_train, weight = weight_tr, free_raw_data = FALSE)

cate_feature <- colnames(x_train %>% select(contains(c("_id", "_flag", "_type")), credit_history, credit_level))
lgb_3 <- lgb.train(params = lgb_param, data = dtrain, nrounds = 1000, categorical_feature = cate_feature)

pred_valid <- predict(lgb_3, data = as.matrix(x_valid))
Meas_b7 <- f1_score(y_pred = ifelse(pred_valid > 0.585, 1, 0), y_true = y_valid, pattern = "macro")




######################################################################################################################
######## Baseline 8: weighting + lgb + classification threshold + feature selection (NULL importance) ################
######################################################################################################################
dtrain <- lgb.Dataset(as.matrix(x_train[feas]), label = y_train, weight = weight_tr, free_raw_data = FALSE)

cate_feature <- intersect(cate_feature, colnames(x_train[feas]))
lgb_4 <- lgb.train(params = lgb_param, data = dtrain, nrounds = 1000, categorical_feature = cate_feature)

pred_valid <- predict(lgb_4, data = as.matrix(x_valid[feas]))
Meas_b8 <- f1_score(y_pred = ifelse(pred_valid > 0.55, 1, 0), y_true = y_valid, pattern = "macro")




## test data
# 1. xgb
dtest <- xgb.DMatrix(as.matrix(car_test[feas]))
pred_test <- predict(xgb.2, newdata = dtest)
submit <- data.frame(customer_id = test$customer_id, loan_default = ifelse(pred_test > 0.25, 1, 0))
write.csv(submit, file = "./res/res2.csv", row.names = FALSE)


# 2. lgb
pred_test <- predict(lgb_1, data = as.matrix(car_test))
submit <- data.frame(customer_id = car_test$customer_id, loan_default = ifelse(pred_test > 0.24, 1, 0))
write.csv(submit, file = "./res/res12_bl_lgb.csv", row.names = FALSE)


package_name <- c("caret", "gbm", "xgboost")
lapply(1:length(package_name), function(i) {if (!require(package_name[i])) install.packages(package_name[i])})



# Model Stacking
first_layer <- function(method, x_train, y_train, x_valid = NULL, x_test, fold = 5) {
  
  train_list <- createFolds(y_train, k = fold, returnTrain = TRUE)
  
  oof_train_pred  <- numeric(nrow(x_train))
  test_pred_summ  <- NULL
  f1_cv <- NULL
  
  if(!is.null(x_valid)) {
    valid_pred_summ <- NULL
  }
  
  for (k in 1:fold) {
    
    print(paste("Running fold:", k))
    
    cv_train_idx <- train_list[[k]]
    cv_valid_idx <- setdiff(1:nrow(x_train), cv_train_idx)
    
    cv_x_train <- x_train[cv_train_idx, ]
    cv_x_valid <- x_train[cv_valid_idx, ]
    cv_y_train <- y_train[cv_train_idx]
    cv_y_valid <- y_train[cv_valid_idx]
    
    
    if (method == "gbm") {
      
      print("train model with gbdt.") 
      
      model <- gbm.fit(x = as.data.frame(cv_x_train), y = cv_y_train, distribution = "bernoulli",
                       n.trees = 500, interaction.depth = 6, n.minobsinnode = 20, shrinkage = 0.05, 
                       nTrain = nrow(cv_x_train))
      
      oof_pred_valid <- predict(model, as.data.frame(cv_x_valid), type = "response")
      pred_test <- predict(model, x_test,  type = "response")
      
      oof_f1_valid <- f1_score(y_pred = ifelse(oof_pred_valid > 0.25, 1, 0), 
                               y_true = cv_y_valid, pattern = "macro")$f1_macro
      
      if(!is.null(x_valid)) {
        pred_valid <- predict(model, x_valid,  type = "response")
      }
    }
    
    
    if (method == "xgb") {
      
      print("train model with xgboost.")
      
      cv_dtrain <- xgb.DMatrix(data = as.matrix(cv_x_train), label = cv_y_train)
      
      cv_dvalid <- xgb.DMatrix(data = as.matrix(cv_x_valid))
      dtest   <- xgb.DMatrix(data = as.matrix(x_test))
      
      param  <- list(eta = 0.01, max_depth = 8, min_child_weight = 1.75, 
                     subsample = 0.87, colsample_bytree = 0.7,  nthread = 4,
                     objective = "binary:logistic", eval_metric = "auc")
      
      model  <- xgb.train(data = cv_dtrain, nrounds = 2000, params = param)
      
      oof_pred_valid <- predict(model, newdata = cv_dvalid)
      pred_test  <- predict(model, newdata = dtest) #, ntreelimit = model$best_ntreelimit
      
      oof_f1_valid <- f1_score(y_pred = ifelse(oof_pred_valid > 0.25, 1, 0), 
                               y_true = cv_y_valid, pattern = "macro")$f1_macro
      
      if(!is.null(x_valid)) {
        dvalid  <- xgb.DMatrix(data = as.matrix(x_valid))
        pred_valid  <- predict(model, newdata = dvalid) 
      }
    }
    
    
    if (method == "lgb") {
      
      print("train model with lightgbm.")
      
      cv_dtrain <- lgb.Dataset(data = as.matrix(cv_x_train), label = cv_y_train)
      
      param  <- list(objective = "binary", boosting = "gbdt", 
                     learning_rate = 0.01, 
                     feature_fraction = 0.7, bagging_fraction = 0.9, 
                     bagging_freq = 5, max_depth = 6, num_leaves = 70, 
                     min_child_weight = 4)
      
      model  <- lgb.train(data = cv_dtrain, params = param, 
                          nrounds = 2000, categorical_feature = cate_feas, 
                          verbose = -1, force_row_wise = TRUE)
      
      oof_pred_valid <- predict(model, data = as.matrix(cv_x_valid))
      pred_test  <- predict(model, data = as.matrix(x_test))
      
      oof_f1_valid <- f1_score(y_pred = ifelse(oof_pred_valid > 0.25, 1, 0), 
                               y_true = cv_y_valid, pattern = "macro")$f1_macro
      
      if(!is.null(x_valid)) {
        pred_test  <- predict(model, data = as.matrix(x_valid))
      }
      
    }
    
    oof_train_pred[cv_valid_idx] <- oof_pred_valid
    f1_cv <- cbind(f1_cv, oof_f1_valid)
    test_pred_summ <- cbind(test_pred_summ, pred_test)
    
    if(!is.null(x_valid)) {
      valid_pred_summ  <- cbind(valid_pred_summ, pred_valid)
    }
    
  }
  
  if(!is.null(x_valid)) {
    res <- list(oof_train_pred = oof_train_pred, 
                valid_pred     = rowMeans(valid_pred_summ),
                test_pred      = rowMeans(test_pred_summ),
                f1_cv          = mean(f1_cv))
  } else {
    res <- list(oof_train_pred, rowMeans(test_pred_summ), mean(f1_cv))
  }

  return (res)
  
}


# sample splitting
set.seed(233)
idx <- createDataPartition(car_train$loan_default, times = 1, p = 0.9, list = FALSE)
x_train <- car_train[idx, ] %>% select(-loan_default, -customer_id)
y_train <- car_train$loan_default[idx]

x_valid <- car_train[-idx, ] %>% select(-loan_default, -customer_id)
y_valid <- car_train$loan_default[-idx]


# construct first-layer model
method <- c("gbm", "xgb", "lgb")
x_train_stack <- x_valid_stack <- x_test_stack <- NULL
f1_cv_stack <- NULL

for (mth in method) {
  
  temp <- first_layer(method = mth, x_train = x.train[feas], y_train = y.train, x_test = car_test[feas])

  x_train_stack <- cbind(x_train_stack, temp[[1]])
  x_valid_stack <- cbind(x_valid_stack, temp[[2]])
  x_test_stack  <- cbind(x_test_stack, temp[[2]])
  
  f1_cv_stack <- cbind(f1_cv_stack, temp[[3]])
  
}

x_train_stack <- data.frame(x_train_stack)
x_test_stack  <- data.frame(x_test_stack)
f1_cv_stack   <- data.frame(f1_cv_stack)

colnames(x_train_stack) <- colnames(f1_cv_stack) <- colnames(x_test_stack) <- method

# x_valid is not NULL
# gbm_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 1] > 0.25, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")
# xgb_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 2] > 0.25, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")
# lgb_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 3] > 0.25, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")

thr_search <- NULL
for (thr in seq(0.235, 0.25, 0.05)) {
  
  gbm_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 1] > thr, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")$f1_macro
  xgb_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 2] > thr, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")$f1_macro
  lgb_Meas <- f1_score(y_pred = ifelse(x_valid_stack[, 3] > thr, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")$f1_macro
  
  thr_search <- cbind(thr_search, c(gbm_Meas, xgb_Meas, lgb_Meas))
}

thr <- list(gbm_thr = seq(0.235, 0.25, 0.05)[which.max(thr_search[1, ])],
            xgb_thr = seq(0.235, 0.25, 0.05)[which.max(thr_search[2, ])],
            lgb_thr = seq(0.235, 0.25, 0.05)[which.max(thr_search[3, ])])

train_stack_label <- sapply(1:3, function(c) {x_train_stack[, c] <- ifelse(x_train_stack[, c] > thr[[c]], 1, 0)})
valid_stack_label <- sapply(1:3, function(c) {x_valid_stack[, c] <- ifelse(x_valid_stack[, c] > thr[[c]], 1, 0)})
test_stack_label  <- sapply(1:3, function(c) {x_test_stack[, c] <- ifelse(x_test_stack[, c] > thr[[c]], 1, 0)})



## sub-level model building 
log_model <- caret::train(x = x_train_stack, y = factor(y.train, levels = c(0, 1)), method = "glm", family = binomial())
log_model <- glm(formula = label ~., data = data.frame(x_train_stack, label = y_train), family = binomial())

pred_valid <- predict(log_model, newdata = x_valid_stack, type = "prob")[, 2]
f1_stack <- f1_score(y_pred = ifelse(pred_valid > 0.25, 1, 0), y_true = y_valid, level_of_label = 2, pattern = "macro")
print(paste("The macro-f1 score of valid data is", unique(f1_stack[,3])))

# [1] "The macro-f1 score of valid data is 0.57758106940538"


pred_test <- predict(log_model, newdata = x_test_stack, type = "prob")[, 2]
submit <- data.frame(customer_id = car_test$customer_id, loan_default = ifelse(pred_test > 0.25, 1, 0))
write.csv(submit, file = "E:/Competition/xf/res/res6_stacking.csv", row.names = FALSE)


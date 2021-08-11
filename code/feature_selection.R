#########################################################################################
################################## function defination ##################################
#########################################################################################

# 1. Coring function uses LightGBM in RandomForest mode fitted on the full dataset
get_feature_importance <- function(data, label, cate_feature, shuffle, seed) {
  
  #' @data: a dataframe or a tibble object.
  #' @label: the name of target variable. vector of labels to use as the target variable
  #' @cate_feature: character vector of column names of catergoricial variables 
  #' @shuffle: bool, whether the label be shuffled or not
  #' @seed: random seed
  
  x_features <- setdiff(colnames(data), label)
  
  # shuffle y if shuffle = TRUE
  if (shuffle) {
    y <- sample(data[[label]], size = nrow(data))
  } else {
    y <- data[[label]]
  }
  
  # fit LightGBM in RF mode
  dtrain <- lgb.Dataset(as.matrix(data %>% select(-all_of(label))), label = y, 
                        categorical_feature = cate_feature, free_raw_data = FALSE)
  
  lgb_param <- list(objective = "binary", boosting = "rf", 
                    num_iterations = 300,  num_threads = 4, 
                    num_leaves = 200, max_depth = 12, 
                    feature_fraction = 0.7, bagging_fraction = 0.7, 
                    bagging_freq = 5, verbose = -1, seed = seed, 
                    force_row_wise = TRUE)
  
  clf <- lgb.train(params = lgb_param, data = dtrain, 
                   categorical_feature = cate_feature)
  imp <- lgb.importance(clf)  
  
  return(imp)
  
}


# 2. calculate the score
get_score <- function(act_imp, null_imp, pattern = c("feature_score", "corr_score")) {
  
  if(is.null(pattern)) pattern <- "corr_score"
  
  if(!pattern %in% c("feature_score", "corr_score")) {
    print("Invalid pattern. Set pattern to 'corr_score'")
    pattern <- "corr_score"
  }
  
  score <- data.frame()
  
  if(pattern == "feature_score") {
    
    for (fea in unique(act_imp$Feature)) {
      gain_f_act_imp  <- mean(act_imp$Gain[act_imp$Feature == fea])
      gain_f_null_imp <- null_imp$Gain[null_imp$Feature == fea]
      
      freq_f_act_imp  <- mean(act_imp$Frequency[act_imp$Feature == fea])
      freq_f_null_imp <- null_imp$Frequency[null_imp$Feature == fea]
      
      gain_score <- log(1 + gain_f_act_imp / (1 + as.numeric(quantile(gain_f_null_imp, 0.75))))
      freq_score <- log(1 + freq_f_act_imp / (1 + as.numeric(quantile(freq_f_null_imp, 0.75))))
      
      score <- rbind(score, c(gain_score, freq_score))
    }
    
    score_df <- cbind(unique(act_imp$Feature), score)
    
  }
  
  if(pattern == "corr_score") {
    
    for (fea in unique(act_imp$Feature)) {
      gain_f_act_imp  <- act_imp$Gain[act_imp$Feature == fea]
      gain_f_null_imp <- null_imp$Gain[null_imp$Feature == fea]
      gain_score <- 100 * sum(gain_f_null_imp < quantile(gain_f_act_imp, 0.25)) / length(gain_f_act_imp)
      
      freq_f_act_imp  <- act_imp$Frequency[act_imp$Feature == fea]
      freq_f_null_imp <- null_imp$Frequency[null_imp$Feature == fea]
      freq_score <- 100 * sum(freq_f_null_imp < quantile(freq_f_act_imp, 0.25)) / length(freq_f_act_imp)
      
      score <- rbind(score, c(gain_score, freq_score))
      
    }
    
    score_df <- cbind(unique(act_imp$Feature), score)
    
  }
  
  colnames(score_df) <- c("feature", "gain_score", "freq_score")
  
  return(score_df)
}


# 3. choose the best combination of variables as the final variables used to build model
score_feature_selection <- function(data, target, train_features, cate_features, method = c("xgb", "lgb")) {
  
  if (method == "lgb") {
    
    # fit lightgbm
    dtrain <- lgb.Dataset(data = as.matrix(data[train_features]), label = target, free_raw_data = FALSE)
    lgb_param <- list(objective = "binary", boosting = "gbdt", learning_rate = 0.05, 
                      num_threads = 4, min_child_weight = 5, metric = "auc",
                      feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 5)
    
    clf <- lgb.cv(params = lgb_param, data = dtrain, nrounds = 1000, nfold = 5, 
                  stratified = TRUE, early_stopping_rounds = 50, 
                  categorical_feature = cate_features, 
                  verbose = 0, force_row_wise = TRUE)
    
    last <- c(clf$record_evals$valid$auc$eval[[clf$best_iter]],
              clf$record_evals$valid$auc$eval_err[[clf$best_iter]])
    
  }
  
  if(method == "xgb") {
    # fit xgboost
    dtrain <- xgb.DMatrix(data = as.matrix(data[train_features]), label = target)
    xgb_param <- list(booster = "gbtree", eta = 0.1, max_depth = 6, min_child_weight = 1.75, 
                      colsample_bytree = 0.7, subsample = 0.85)
    
    clf <- xgb.cv(data = dtrain, params = xgb_param, nrounds = 1000, nfold = 5, 
                  metrics = "auc", nthreads = 2, objective = "binary:logistic")
  
    last <- c(clf$evaluation_log[nrow(clf$evaluation_log), "test_auc_mean"], 
              clf$evaluation_log[nrow(clf$evaluation_log), "test_auc_std"])
  
  }

  return(last)
}


### 4. visualization function
display_distributions <- function(act_imp_df, null_imp, feature) {
  
  # Plot Split and Gain importances
  spImp <- ggplot(data = tibble(Freq_null = null_imp$Frequency[null_imp$Feature == feature], 
                                Freq_act = act_imp_df$Frequency[act_imp_df$Feature == feature])) +
    geom_histogram(aes(x = Freq_null)) +
    geom_vline(aes(xintercept = Freq_act), color = "red", width = 12) +
    ggtitle(paste("Split Importance of", feature)) +
    xlab(paste("Null Importance Distribution for ", feature))

  gnImp <- ggplot(data = tibble(gain_null = null_imp$Gain[null_imp$Feature == feature], 
                                gain_act = act_imp_df$Gain[act_imp_df$Feature == feature])) +
    geom_histogram(aes(x = gain_null)) +
    geom_vline(aes(xintercept = gain_act), color = "red", width = 12) +
    ggtitle(paste("Gain Importance of", feature)) +
    xlab(paste("Null Importance Distribution for ", feature))
  
  # plot <- ggarrange(spImp, gnImp, ncol = 2, nrow = 1)
  return(list(spImp, gnImp))
   
}



#########################################################################################
################################### feature selection ###################################
#########################################################################################
cate_feature <- car_train %>% select(-customer_id) %>% 
  select(contains(c("_id", "_flag", "_type", "credit_history", "credit_level"))) %>% colnames()

iterations <- 50

# Step 1: get two version of feature importance 
act_imp_df <- shf_imp_df <- NULL

for (iter in 1:iterations) {
  temp_act <- get_feature_importance(data = car_train %>% select(-customer_id), 
                                     label = "loan_default", 
                                     cate_feature = cate_feature, 
                                     shuffle = FALSE, seed = 311)
  temp_shf <- get_feature_importance(data = car_train %>% select(-customer_id), 
                                     label = "loan_default", 
                                     cate_feature = cate_feature, 
                                     shuffle = TRUE, seed = 233)
  
  temp_act$run <- iter
  temp_shf$run <- iter
  
  act_imp_df <- rbind(act_imp_df, temp_act)
  shf_imp_df <- rbind(shf_imp_df, temp_shf)
  
  cat("Done with", iter, "of", iterations, "\n")
  
}


# calculate the score
feature_score <- get_score(act_imp = act_imp_df, null_imp = shf_imp_df, pattern = "feature_score")
corr_score <- get_score(act_imp = act_imp_df, null_imp = shf_imp_df, pattern = "corr_score")



sink(file = "E:/Competition/xf/output/thrs_fs_eval.txt")
for (threshold in c(0, 10, 20, 80, 90)) {
  
  gain_feas     <- unique(corr_score$feature[corr_score$gain_score >= threshold])
  gain_cat_feas <- unique(corr_score$feature[corr_score$gain_score >= threshold & 
                                               corr_score$feature %in% cate_feature])
  
  freq_feas     <- unique(corr_score$feature[corr_score$freq_score >= threshold])
  freq_cat_feas <- unique(corr_score$feature[corr_score$freq_score >= threshold & 
                                               corr_score$feature %in% cate_feature])
  
  cat("Results for threshold = ", threshold, "\n")
  gain_results <- score_feature_selection(data = car_train, target = car_train[["loan_default"]], 
                                          train_features = gain_feas, cate_features = gain_cat_feas, 
                                          method = "lgb")
  cat("\t GAIN : ", gain_results[1], "+/-", gain_results[2], "\n")
  
  
  freq_results <- score_feature_selection(data = car_train, target = car_train[["loan_default"]], 
                                          train_features = freq_feas, cate_features = freq_cat_feas,
                                          method = "lgb")
  cat("\t SPLIT : ", freq_results[1], "+/-", freq_results[2], "\n")
  
}

sink()


threshold <- 0
freq_feas_0     <- unique(corr_score$feature[corr_score$freq_score > threshold])
freq_cat_feas_0 <- unique(corr_score$feature[corr_score$freq_score > threshold & 
                                                corr_score$feature %in% cate_feature])

sink(file = "E:/Competition/xf/output/feature_selection.txt")
for (threshold in c(0, 10, 20, 40, 90)) {
  cat("For threshold =", threshold, "\n")
  print(unique(corr_score$feature[corr_score$freq_score >= threshold]))
}
sink()


# Visualization 
total_inact_loan <- display_distributions(act_imp = act_imp_df, null_imp = shf_imp_df, feature = "total_inactive_loan_no")
ggarrange(total_inact_loan[[1]], total_inact_loan[[2]], ncol = 2, nrow = 1)

 


# The Way 1.0 to calculate score (non-functional)
{ 
  # # Method 1: score = 未shuffle的特征重?????? / shuffle特征重要型的75%分位???
  # feature_score <- NULL
  # 
  # for (fea in act_imp_df$Feature) {
  #   gain_f_act_imp <- mean(act_imp_df$Gain[act_imp_df$Feature == fea])
  #   gain_f_shf_imp <- shf_imp_df$Gain[shf_imp_df$Feature == fea]
  #   
  #   freq_f_act_imp <- mean(act_imp_df$Frequency[act_imp_df$Feature == fea])
  #   freq_f_shf_imp <- shf_imp_df$Frequency[shf_imp_df$Feature == fea]
  #   
  #   gain_score <- log(1 + gain_f_act_imp / (1 + as.numeric(quantile(gain_f_shf_imp, 0.75))))
  #   freq_score <- log(1 + freq_f_act_imp / (1 + as.numeric(quantile(freq_f_shf_imp, 0.75))))
  # 
  #   feature_score <- append(feature_score, c(fea, gain_score, freq_score))
  # }
  # 
  # score_df <- as.data.frame(matrix(feature_score, length(act_imp_df$Feature), 3, byrow = TRUE))
  # colnames(score_df) <- c("feature", "gain_score", "freq_score")
  # 
  # 
  # 
  # # Method 2: shuffle之后特征重要性低于实际target对应特征重要???25%分位数的次数，所占百分比
  # correlation_score <- NULL
  # for (fea in act_imp_df$Feature) {
  #   
  #   gain_f_act_imp <- act_imp_df$Gain[act_imp_df$Feature == fea]
  #   gain_f_shf_imp <- shf_imp_df$Gain[shf_imp_df$Feature == fea]
  #   gain_score <- 100 * sum(gain_f_shf_imp < quantile(gain_f_act_imp, 0.25)) / length(gain_f_act_imp)
  #   
  #   freq_f_act_imp <- act_imp_df$Frequency[act_imp_df$Feature == fea]
  #   freq_f_shf_imp <- shf_imp_df$Frequency[shf_imp_df$Feature == fea]
  #   freq_score <- 100 * sum(freq_f_shf_imp < quantile(freq_f_act_imp, 0.25)) / length(freq_f_act_imp)
  #  
  # 
  #   correlation_score <- append(correlation_score, c(fea, gain_score, freq_score)) 
  # }
  
  # score_df_corr <- as.data.frame(matrix(correlation_score, length(act_imp_df$Feature), 3, byrow = TRUE))
  # colnames(score_df_corr) <- c("feature", "gain_score", "freq_score")
}

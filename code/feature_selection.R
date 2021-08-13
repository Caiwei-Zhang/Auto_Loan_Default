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





# 
# # Visualization 
# total_inact_loan <- display_distributions(act_imp = act_imp_df, null_imp = shf_imp_df, feature = "total_inactive_loan_no")
# ggarrange(total_inact_loan[[1]], total_inact_loan[[2]], ncol = 2, nrow = 1)
# 
#  



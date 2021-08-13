# read dataset
setwd("E:/Competition/xf/Auto_Loan_Default")
train <- read_csv(file = "./data/train.csv") %>% as_tibble()
test  <- read_csv(file = "./data/test.csv") %>% as_tibble()
car   <- rbind(select(train, -loan_default), test)


################################## Intro: View the data roughly ##############################
# the structure of car 
str(car)
car$outstanding_disburse_ratio <- as.numeric(car$outstanding_disburse_ratio)

# view the data with histogram
length(table(car$branch_id))
table(car$supplier_id) 
hist(car$asset_cost[car$asset_cost < quantile(car$asset_cost, 0.99)])
hist(car$loan_to_asset_ratio)

par(mfrow = c(1,3))
hist(car$outstanding_disburse_ratio, main = "Histogram of ODR (merged)")
hist(car$outstanding_disburse_ratio[1:150000], main = "Histogram of ODR (trian)")
hist(car$outstanding_disburse_ratio[150001:180000], main = "Histogram of ODR (test)")




### Task 1: distinguish categoricial variables 
col <- car %>% select(-customer_id) %>% 
  select(contains(c("_id", "_flag", "_type")), Credit_level) %>% colnames()

sink(file = "./output/fct_level_table.txt")
for (c in col) {
  cat(c, length(table(car[[c]])), "\n")
  
  if(length(table(car[[c]])) <= 100) {
    temp <- data.frame(table(car[[c]]))
    colnames(temp) <- c("LEVELS", "COUNT")
    print(temp)
  }
}
sink()

# drop variables with 1 level and employee_code_id
temp <- NULL
for (c in col) temp <- append(temp, length(table(car[[c]])))
fct_level <- data.frame(colname = col, level = temp)
car <- car %>% select(-all_of(fct_level[fct_level$level == 1, 1])) %>%
  select(-employee_code_id)




### Task 2: date columns
date_check <- car %>% select(year_of_birth, age, disbursed_date) %>% 
  mutate(gap_age = 2021 - year_of_birth - age, 
         gap_disburse = 2021 - disbursed_date)
table(date_check$gap_age)
table(date_check$gap_disburse)

car <- car %>% select(-year_of_birth, -disbursed_date)




### Task 3: the relationships between columns containing "main","sub" and "total  
com_pattern <- c("account_loan_no", "inactive_loan_no", "overdue_no", "active_loan_no",
                 "outstanding_loan", "sanction_loan", "disbursed_loan", "monthly_payment", "tenure")
for (p in com_pattern) {
  if(sum(str_detect(colnames(car), paste("total", p, sep = "."))) == 0) {
    cat("Missing feature of total account:", p, "\n")
  } else { # 可以进行加和
    temp <- sum(car %>% select(contains(p) & contains("main")) + 
                  car %>% select(contains(p) & contains("sub")) 
                != car %>% select(contains(p) & contains("total")))
    
    if (temp > 0) {
      cat("\nThe relationship between variables containing '", p, 
          "' is not summation. With", temp, "abnormal values. \n")}
  }
}

car <- car %>% mutate(total_inactive_loan_no = main_account_inactive_loan_no + sub_account_inactive_loan_no,
                      total_active_loan_no = main_account_active_loan_no + sub_account_active_loan_no,
                      total_account_tenure = main_account_tenure + sub_account_tenure) 




### Task 4: NA check & infinite check 
summ_car <- summarizeColumns(car %>% select(-customer_id))

# outstanding_disburse_ratio contains -Inf and NA. Fill with median.
car$outstanding_disburse_ratio[is.na(car$outstanding_disburse_ratio)] <-
  median(car$outstanding_disburse_ratio, na.rm = TRUE)
car$outstanding_disburse_ratio[is.infinite(car$outstanding_disburse_ratio)] <- 
  median(car$outstanding_disburse_ratio, na.rm = TRUE)

# whether NA exists 
colnames(car)[colSums(is.na(car)) > 0]




### Task 5: chenck the variables with zero or near-zero variance 
nzv <- nearZeroVar(car, freqCut = 95/5, saveMetrics = TRUE)
nzv




### Task 6: check columns with negative values (wrong data or with implication) 
car <- car %>% mutate(credit_level = ifelse(Credit_level < 0, 0, Credit_level)) %>% select(-Credit_level)
# fea_with_neg <- summ_car$name[summ_car$min < 0]
# neg_prop <- percent(colSums(car %>% select(all_of(fea_with_neg)) < 0) / nrow(car), 0.01)

# op 1: delete thhe rows with negetive values.
# car <- car %>% mutate(credit_level = ifelse(Credit_level < 0, 0, Credit_level)) %>% select(-Credit_level) %>% 
#   filter(main_account_outstanding_loan >= 0, sub_account_outstanding_loan >= 0,
#          total_outstanding_loan >= 0, outstanding_disburse_ratio >= 0)

# op 2: substitute the negtive values with 0.
# car <- car %>% mutate(credit_level = ifelse(Credit_level < 0, 0, Credit_level)) %>% select(-Credit_level)
#                       main_account_outstanding_loan = ifelse(main_account_outstanding_loan < 0, 0, main_account_outstanding_loan),
#                       sub_account_outstanding_loan  = ifelse(sub_account_outstanding_loan < 0, 0, sub_account_outstanding_loan),
#                       total_outstanding_loan        = ifelse(total_outstanding_loan < 0, 0, total_outstanding_loan),
#                       outstanding_disburse_ratio    = ifelse(outstanding_disburse_ratio  < 0, 0, outstanding_disburse_ratio))
#                       colSums(data < 0)




### Task 7: correlation analysis 
mt <- round(cor(car), 2)
corr_plot <- corrplot(corr = mt, method = "shade", type = "full", tl.col = "black")

for (c in 1:(ncol(mt) - 1)) {
  for (r in (c+1):nrow(mt)) {
    if (mt[r, c] >= 0.8) cat("The correlation of [", colnames(car)[r], "] and [", 
                             colnames(car)[c], "] is", mt[r, c], "\n")
  }
}



### Task 8: feature construction
car <- car %>% mutate(person_info_flag = Driving_flag + passport_flag) %>%
  select(-Driving_flag, -passport_flag)
 # age_splt = ifelse(age <= 25, 1, ifelse(age <= 30, 2, ifelse(age < 40, 3, 4))))
write.csv(car, file = "./output/car.csv")


### Task 9-1: splitting train data & test data  (data normalizing)
cate_feature <- car %>% select(-customer_id) %>% 
  select(contains(c("_id", "_flag", "_type")), credit_level) %>% colnames()
cont_feature <- setdiff(colnames(car), c(cate_feature, "customer_id"))

preProc <- preProcess(car[, cont_feature], method = "range", rangeBounds = c(0, 1))
car_train <- predict(preProc, car[1:150000, cont_feature]) %>% 
  mutate(loan_default = train$loan_default) %>% 
  cbind(car[1:150000, cate_feature]) %>% as_tibble()
car_test <- predict(preProc, car[150001:180000, cont_feature]) %>% 
  cbind(car[1:150000, cate_feature]) %>% as_tibble()

## Task 9-2: splitting train data & test data 
car_train <- car[1:150000, ] %>% select(-customer_id) %>%
  mutate(loan_default = train$loan_default)
car_test <- car[150001:180000, ] %>% select(-customer_id)



### Task 10: outliers check 

# cont_feature
c1 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = disbursed_amount)) + 
  geom_boxplot() + theme_bw() # outlier
c2 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = asset_cost)) + 
  geom_boxplot() + theme_bw() # outlier
c3 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = credit_score)) + 
  geom_boxplot() + theme_bw()
c4 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = last_six_month_new_loan_no)) + 
  geom_boxplot() + theme_bw()
c5 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = last_six_month_defaulted_no)) + 
  geom_boxplot() + theme_bw()
c6 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = average_age)) + 
  geom_boxplot() + theme_bw()
c7 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = credit_history)) + 
  geom_boxplot() + theme_bw()
c8 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = enquirie_no)) + 
  geom_boxplot() + theme_bw()
c9 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = loan_to_asset_ratio)) + 
  geom_boxplot() + theme_bw()
c10 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = total_account_loan_no)) + 
  geom_boxplot() + theme_bw() # outlier
c11 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = total_inactive_loan_no)) + 
  geom_boxplot() + theme_bw() # outlier
c12 <- ggplot(data = car_train, mapping = aes(x = factor(loan_default), y = total_overdue_no)) + 
  geom_boxplot() + theme_bw()

ggarrange(c1, c2, c3, c4, c5, c6, ncol = 3, nrow = 2)
ggarrange(c7, c8, c9, c10, c11, c12, ncol = 3, nrow = 2)

# remove outliers
o1 <- which(car_train$disbursed_amount > quantile(car_train$disbursed_amount, 0.9999))
o2 <- which(car_train$asset_cost > quantile(car_train$asset_cost, 0.9999))
o3 <- which(car_train$total_account_loan_no > quantile(car_train$total_account_loan_no, 0.9999))
o4 <- which(car_train$total_inactive_loan_no > quantile(car_train$total_inactive_loan_no, 0.9999))
rm.idx <- unique(c(o1, o2, o3, o4))

car_train <- car_train[-rm.idx, ]


# cate_featrue
a <- ggplot(data = car_train, mapping = aes(x = branch_id, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()
b <- ggplot(data = car_train, mapping = aes(x = supplier_id, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()
c <- ggplot(data = car_train, mapping = aes(x = manufacturer_id, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()
d <- ggplot(data = car_train, mapping = aes(x = area_id, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()
e <- ggplot(data = car_train, mapping = aes(x = employment_type, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()
f <- ggplot(data = car_train, mapping = aes(x = credit_level, fill = factor(loan_default))) + 
  geom_bar() + theme_bw()

ggarrange(a, b, c, f, e, f, ncol = 3, nrow = 2)




#### Task 11: feature selection
source("./code/feature_selection.R")
iterations <- 20

# Step 1: get two version of feature importance 
act_imp_df <- shf_imp_df <- NULL

for (iter in 1:iterations) {
  temp_act <- get_feature_importance(data = car_train, 
                                     label = "loan_default", 
                                     cate_feature = cate_feature, 
                                     shuffle = FALSE, seed = 311)
  temp_shf <- get_feature_importance(data = car_train, 
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


for (threshold in seq(0, 99, 10)) {
  
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


threshold <- 30
freq_feas_30     <- unique(corr_score$feature[corr_score$freq_score >= threshold])
freq_cat_feas_30 <- unique(corr_score$feature[corr_score$freq_score >= threshold & 
                                               corr_score$feature %in% cate_feature])

sink(file = "./output/feature_selection.txt")
for (threshold in seq(0, 99, 10)) {
  cat("For threshold =", threshold, "\n")
  print(unique(corr_score$feature[corr_score$freq_score >= threshold]))
}
sink()






### Visualization
sum(car_train$Driving_flag[car_train$loan_default == 0] == 1) / sum(car_train$Driving_flag[car_train$loan_default == 0] == 0)
sum(car_train$Driving_flag[car_train$loan_default == 1] == 1) / sum(car_train$Driving_flag[car_train$loan_default == 1] == 0)

sum(car_train$passport_flag[car_train$loan_default == 0] == 1) / sum(car_train$passport_flag[car_train$loan_default == 0] == 0)
sum(car_train$passport_flag[car_train$loan_default == 1] == 1) / sum(car_train$passport_flag[car_train$loan_default == 1] == 0)

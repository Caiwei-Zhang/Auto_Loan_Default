
# read dataset
setwd("E:/Competition/xf/data")
train <- read_csv(file = "./train.csv") %>% as_tibble()
test  <- read_csv(file = "./test.csv") %>% as_tibble()
car <- rbind(select(train, -loan_default), test)


##############################################################################################
################################## Intro: View the data roughly ##############################
##############################################################################################
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



##############################################################################################
############################# Task 1: distinguish  categoricial variables ####################
##############################################################################################
col <- colnames(car %>% select(contains(c("_id", "_flag", "_type")), credit_history, Credit_level))
sink(file = "./fct_level_table.txt")
for (c in col) {
  cat(c, length(table(car[[c]])), "\n")
  
  if(length(table(car[[c]])) <= 100) {
    temp <- data.frame(table(car[[c]]))
    colnames(temp) <- c("LEVELS", "COUNT")
    print(temp)
  }
}
sink()

# drop variables with 1 level.
temp <- NULL
for (c in col) temp <- append(temp, length(table(car[[c]])))
fct_level <- data.frame(colname = col, level = temp)
car <- car %>% select(-all_of(fct_level[fct_level$level == 1, 1]))

# drop irrelevant categoricial varaibles with thousands of levels
car <- car %>% select(-c(supplier_id, employee_code_id))



##############################################################################################
###################### Task 2: cross reference of [year_of_birth] and [age] ##################
##############################################################################################
age_year <- car %>% select(year_of_birth, age) %>% mutate(age_0 = 2021 - year_of_birth)
sum((2021 - car$year_of_birth - 2) != car$age) # [1] 0
car <- car %>% select(-year_of_birth)



##############################################################################################
########### Task 3: the relation between variables start with "main_","sub" and "total #######
##############################################################################################
com_pattern <- c("account_loan_no", "inactive_loan_no", "overdue_no", 
                 "outstanding_loan", "sanction_loan", "disbursed_loan", "monthly_payment")
for (p in com_pattern) {
  temp <- sum(car %>% select(contains(p)) %>% select(starts_with("main")) + 
                car %>% select(contains(p)) %>% select(starts_with("sub")) 
              != car %>% select(contains(p)) %>% select(starts_with("total")))
  
  if (temp > 0)  print(paste(p, temp, sep = ":"))
}

# inactive_loan_no
inactive_loan_no <- car %>% select(contains("inactive_loan_no")) %>% 
  mutate(ref = sub_account_inactive_loan_no + main_account_inactive_loan_no) 

# revise the value of [total_inactive_loan_no]
car <- car %>% mutate(total_inactive_loan_no = main_account_inactive_loan_no + sub_account_inactive_loan_no)




##############################################################################################
############################### Task 4: convert date-type into interval ######################
##############################################################################################
car <- car %>% mutate(disbursed_interval = 2021 - disbursed_date)
table(car$disbursed_interval)
car <- car %>% select(-disbursed_date, -disbursed_interval)




##############################################################################################
##################################### Task 5: NA check & outlier check #######################
##############################################################################################
summ_car <- summarizeColumns(car %>% select(-customer_id))

# outstanding_disburse_ratio contains -Inf and NA. Fill with median.
car$outstanding_disburse_ratio[is.na(car$outstanding_disburse_ratio)] <-
  median(car$outstanding_disburse_ratio, na.rm = TRUE)
car$outstanding_disburse_ratio[is.infinite(car$outstanding_disburse_ratio)] <- 
  median(car$outstanding_disburse_ratio, na.rm = TRUE)

# whether NA exists 
col_na_tab <- colSums(is.na(car))
col_na_tab[col_na_tab > 0]




##############################################################################################
#################### Task 6: chenck the variables with zero or near-zero variance ############
##############################################################################################
nzv <- nearZeroVar(car, freqCut = 95/5, saveMetrics = TRUE)
nzv


##############################################################################################
######### Task 7: check the variables with negative values (wrong data or implicate sth) #####
##############################################################################################
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
# 

write.csv(car, file = "./car_train_test.csv")



# split the train dataset and test dataset
car_train <- car[1:150000, ] %>% mutate(loan_default = train[["loan_default"]])
car_test  <- car[150001:180000, ]



######### branch_id 
ggplot(data = cbind(car, is_train = c(rep(1, 150000), rep(0, 30000))), 
                    mapping = aes(x = branch_id, fill = factor(is_train))) + 
  geom_histogram() + facet_grid(is_train ~ .) + theme_bw()

sum(abs(prop.table(table(car_train$branch_id)) - prop.table(table(car_test$branch_id))))


######## area_id
ggplot(data = cbind(car, is_train = c(rep(1, 150000), rep(0, 30000))), 
       mapping = aes(x = area_id, fill = factor(is_train))) + 
  geom_bar() + 
  facet_grid(is_train ~ .) + theme_bw()

ggplot(data = car_train, mapping = aes(x = branch_id, fill = factor(loan_default))) + 
  geom_histogram() + 
  facet_grid(loan_default ~ .) + theme_bw()


####### manufacturer_id
ggplot(data = cbind(car, is_train = c(rep(1, 150000), rep(0, 30000))), 
       mapping = aes(x = manufacturer_id, fill = factor(is_train))) + 
  geom_bar() + 
  facet_grid(is_train ~ .) + theme_bw()

ggplot(data = car_train, mapping = aes(x = manufacturer_id, fill = factor(loan_default))) + 
  geom_bar() + 
  facet_grid(loan_default ~ .) + theme_bw()


####### employment_type
ggplot(data = cbind(car, is_train = c(rep(1, 150000), rep(0, 30000))), 
       mapping = aes(x = employment_type, fill = factor(is_train))) + 
  geom_bar() + 
  facet_grid(is_train ~ .) + theme_bw()

ggplot(data = car_train, mapping = aes(x = employment_type, fill = factor(loan_default))) + 
  geom_bar() + 
  facet_grid(loan_default ~ .) + theme_bw()


######### main_account_outstanding_loan
tdata <- tibble(is_neg = ifelse(car$main_account_outstanding_loan < 0, 1, 0), 
                is_train = c(rep(1, 150000), rep(0, 30000)))
ggplot(data = tdata, mapping = aes(x = factor(is_neg), fill = factor(is_train))) + 
  geom_bar() + theme_bw()

# the negtive value proportion of train dataset and test dataset
sum(car_train$main_account_outstanding_loan < 0, 1, 0) / sum(car_train$main_account_outstanding_loan >= 0, 1, 0) 
# [1] 0.002017368
sum(car_test$main_account_outstanding_loan < 0, 1, 0) / sum(car_test$main_account_outstanding_loan >= 0, 1, 0) 
# [1] 0.00197041


######### sub_account_outstanding_loan
sum(car_train$sub_account_outstanding_loan < 0, 1, 0) / sum(car_train$sub_account_outstanding_loan >= 0, 1, 0) 
# [1] 0.0002400544
sum(car_test$sub_account_outstanding_loan < 0, 1, 0) / sum(car_test$sub_account_outstanding_loan >= 0, 1, 0) 
# 0.0003334222

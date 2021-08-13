
packages <- c("devtools", "tidyverse", "caret", "mlr",
              "xgboost", "randomForest", "ROSE", "scales",
              "ggplot2", "ggpubr", "gridExtra", "corrplot", 
              "glmnet", "gbm")

for (pkg in packages) {
  if (!require(pkg)) {install.packages(pkg)}
}

# install LightGBM (recommend install lightgbm with  CMake and Visual Studio)
install.packages("lightgbm", repos = "https://cran.r-project.org")

library(lightgbm)
library(tidyverse)
library(mlr)
library(corrplot)
library(xgboost)
library(caret)

library(ROSE)
library(ggplot2)
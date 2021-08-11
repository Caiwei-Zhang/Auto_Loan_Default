# library()
packages <- c("devtools", "tidyverse", "caret", 
              "xgboost", "randomForest", "ROSE", "scales",
              "ggplot2", "ggpubr", "gridExtra")

for (pkg in packages) {
  if (!require(pkg)) {install.packages(pkg)}
}

# install LightGBM
install.packages("lightgbm", repos = "https://cran.r-project.org")
install.packages("lightgbm", type = "both", repos = "https://cran.r-project.org")

## install CatBoost
install_url("https://github.com/catboost/catboost/releases/download/v0.16.5/catboost-R-Windows-0.16.5.tgz", 
            INSTALL_opts = c("--no-multiarch", "--no-test-load"))

library(lightgbm)
library(tidyverse)
library(ggplot2)
library(ggpubr)

library(caret)
library(ROSE)
library(mlr)
library(scales)
library(xgboost)
library(catboost)
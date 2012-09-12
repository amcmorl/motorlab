## Load all the functions, simulate, fit and plot
## zhanwu Liu Jan 29, 2010

## There are many ways to run this:
# (1) copy + paste this code
# (2) In the R console, type source("run_test.R")
# (3) In Linux console prompt, type R CMD BATCH run_test.R


## 0. Load all the functions
# you may need to use "setwd("dirname")" to set working directory
# "getwd()" will tell you the current working directory


source("common_functions.R")
source("simulate_realkinematics.R")
source("fit_CV_allback.R")

## 3. Simulate from real kinematics with small noise, fit with CV, note that the different format of returned output
test = simu.realpos(sd.factor=0.0001)
res = fit_CV_allback(test)
sse = res[0]
predict = res[1]




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
source("simulate_GaussianSpeed_8direction.R")
source("simulate_realkinematics.R")
source("fit_noCV.R")
source("fit_CV.R")


## 1. Simulate from Gaussian speed profile with small noise, fit without CV
test1 = simu.8d(sd.factor=0.0001)
out1 = fit_noCV(test1)

# plot out to a postscript file
postscript("test1.ps",width=6,height=6,horizontal=FALSE)
par("las"=2) #choose direction of axis label text so that all can be displayed
image(1:16,1:16, t((out1$aic)),col=heat.colors(20),xlab="Decoding",ylab="Encoding",axes=FALSE)

modellist = c("kd","kdp","kds","kdps","kv","kvp","kvs","kvps","kdX","kdpX","kdsX","kdpsX","kvX","kvpX","kvsX","kvpsX")
 axis(1, at = 1:16, label = modellist,cex=1)
 axis(2, at = 1:16, label = modellist,cex=1)
dev.off()


## 2. Simulate from real kinematics with small noise, fit without CV
test2 = simu.realpos(sd.factor=0.0001)
out2 = fit_noCV(test2)

# plot out to a postscript file
postscript("test2.ps",width=6,height=6,horizontal=FALSE)
par("las"=2) #choose direction of axis label text so that all can be displayed
image(1:16,1:16, t((out2$aic)),col=heat.colors(20),xlab="Decoding",ylab="Encoding",axes=FALSE)

modellist = c("kd","kdp","kds","kdps","kv","kvp","kvs","kvps","kdX","kdpX","kdsX","kdpsX","kvX","kvpX","kvsX","kvpsX")
 axis(1, at = 1:16, label = modellist,cex=1)
 axis(2, at = 1:16, label = modellist,cex=1)
dev.off()

## 3. Simulate from real kinematics with small noise, fit with CV, note that the different format of returned output
test3 = simu.realpos(sd.factor=0.0001)
out3 = fit_CV(test3)

# Find out which model fits best for each dataset
min.model = rep(0,16)
for ( j in 1:16) {
	min.model[j] = which.min(apply(out3[[j]],2, sum))
}
print(min.model)

# plot out the CV result of the first dataset to a postscript file
postscript("test3.1.ps",width=6,height=6,horizontal=FALSE)
par("las"=2) #choose direction of axis label text so that all can be displayed
image(1:16,1:10, t(log(out3[[1]])),col=heat.colors(20),xlab="Decoding",ylab="Repeats",axes=FALSE)

modellist = c("kd","kdp","kds","kdps","kv","kvp","kvs","kvps","kdX","kdpX","kdsX","kdpsX","kvX","kvpX","kvsX","kvpsX")
 axis(1, at = 1:16, label = modellist,cex=1)
 axis(2, at = 1:10, label = 1:10,cex=1)
dev.off()




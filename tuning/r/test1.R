source("common_functions.R")
source("simulate_GaussianSpeed_8direction.R")
source("simulate_realkinematics.R")
source("fit_noCV.R")
source("fit_CV.R")


## 1. Simulate from Gaussian speed profile with small noise, fit without CV
test1 = simu.8d(sd.factor=0.0000)
out1 = fit_noCV(test1)

# plot out to a postscript file
postscript("test1.ps",width=6,height=6,horizontal=FALSE)
par("las"=2) #choose direction of axis label text so that all can be displayed
image(1:16,1:16, t((out1$aic)),col=heat.colors(20),xlab="Decoding",ylab="Encoding",axes=FALSE)

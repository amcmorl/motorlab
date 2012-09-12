source("common_functions.R")
source("simulate_GaussianSpeed_8direction.R")
source("simulate_realkinematics.R")
source("fit_noCV.R")
source("fit_CV.R")

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

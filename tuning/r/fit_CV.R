## Provides function:  fit_CV
## Given simulated data, randomly take half data, fit and predict in another half
##     repeat n.cv times and report the SSE from each random draw
## zhanwu Liu Jan 28, 2010

## Input: a simulated or real dataset (assume object name is "simdata")
## The format is: 
##      simdata$firing.rate  The matrix of firing rate, each column corresponds to the firing
##                           rate simulated from one model
##      simdata$Xmat         The matrix contains simulated kinematics information
##                           columns: 1 time, 2-4 direction, 5-7 position, 8 speed, 
##                                    9-11 velocity, 12 target/trial id
##                                    for fitting only position should be used

## Returns: let n.dataset be the number of sets of firing rates, 
##              n.cv be the number of cross-validation draws, return the following:
## One list object contains n.dataset matrices, each matrix is n.cv times 16
##     e.g. each row is from one random draw, calculate the sse from fitting all 16 models

source('common_functions.R')

fit_CV = function (moddata, n.cv=10) {
  moddata$Xmat[,c(2:4,8:11)] = 0
  require(mgcv)
  ## The value for returning
  SSE.MAT.ALL = list(NULL)
  ## how many sets of firing rate from same set of 
  ## kinematics information 
  ## And initiate the matrix used to store results
  n.dataset = dim(moddata$firing.rate)[2]
  
  for (j in 1:n.dataset) {
    # Assign values from input to facilitate processing
	yvec = moddata$firing.rate[,j]
	Xmat.temp = moddata$Xmat
	
	# Remove the first and last time point for each trial
	ind.remove = which(duplicated(Xmat.temp[,12])==FALSE) 
	ind.remove=sort(c(ind.remove[-1]-1,ind.remove, length(moddata$firing.rate[,j])))

	yvec = yvec[-ind.remove]

	#reconstruct speed in the matrix
	trials.id  = unique(Xmat.temp[,12])
	Xmat = numeric(0)
	for ( i in 1:length(trials.id)) {
      ind.trial = which(Xmat.temp[,12]==trials.id[i])
      Xmat.trial = Xmat.temp[ind.trial,]
      pos.trial = Xmat.trial[,c(5:7)]
		
      # start here to toggle comments
      vel.trial = matrix(0,length(pos.trial[,1])-1, 3)
      vel.trial[,1] = diff(pos.trial[,1])
      vel.trial[,2] = diff(pos.trial[,2])
      vel.trial[,3] = diff(pos.trial[,3])
		
      speed.trial = apply(vel.trial, 1, vecnorm)
      
      Xmat.trial = Xmat.trial[-1,] #remove first point
      Xmat.trial[,c(9:11)]=vel.trial
      Xmat.trial[,8]=speed.trial
		
		
      #Calculate direction 
      for ( ii in 1:length(vel.trial[,1])) {
        Xmat.trial[ii,c(2:4)] = vel.trial[ii,]/speed.trial[ii]
      }
		
      #Remove the last point
      Xmat.trial = Xmat.trial[-length(Xmat.trial[,1]),]
		
      Xmat=rbind(Xmat, Xmat.trial)
	}

    # the matrix to hold the cross-validation sse
	cv.sse.mat = matrix(0, n.cv, 16)	
	
	colnames(Xmat) = c("t","dx","dy","dz","px","py","pz","speed","vx","vy","vz","tid")
	ALLDATA = as.data.frame(cbind(yvec,Xmat))
	
	for ( iiii  in 1:n.cv) {
      ind.train=sort(sample(1:length(yvec),length(yvec)/2))
      ind.test = c(1:length(yvec))[-ind.train]
      
      DATA1 = ALLDATA[ind.train,]
      DATA2 = ALLDATA[ind.test,]	
	
      #fit for each model
      #kd 
      out.kd = gam(yvec~dx+dy+dz,data=DATA1)
	
      #kdp 
      out.kdp = gam(yvec~dx+dy+dz+px+py+pz,data=DATA1)
	
	  #kds 
      out.kds = gam(yvec~dx+dy+dz+speed, data=DATA1)
	
	  #kdps 
      out.kdps = gam(yvec~dx+dy+dz+px+py+pz+speed, data=DATA1)
	
      #kv
      out.kv = gam(yvec~vx+vy+vz, data=DATA1)
	
  	  #kvp 
      out.kvp = gam(yvec~vx+vy+vz+px+py+pz, data=DATA1)
	
	  #kvs 
      out.kvs = gam(yvec~vx+vy+vz+speed, data=DATA1)
	
	  #kvps 
      out.kvps = gam(yvec~vx+vy+vz+px+py+pz+speed, data=DATA1)
	
	  #kdX
      out.kdX = gam(yvec~ s(t, by=dx)+s(t,by=dy)+s(t,by=dz), data=DATA1)
	
	  #kdpX
      out.kdpX = gam(yvec~ s(t, by=dx)+s(t,by=dy)+s(t,by=dz)+px+py+pz, data=DATA1)
	
	  #kdsX
      out.kdsX = gam(yvec~ s(t, by=dx)+s(t,by=dy)+s(t,by=dz)+speed, data=DATA1)
		
	  #kdpsX
      out.kdpsX = gam(yvec~ s(t, by=dx)+s(t,by=dy)+s(t,by=dz)+px+py+pz+speed, data=DATA1)
	
	  #kvX
      out.kvX = gam(yvec~ s(t, by=vx)+s(t,by=vy)+s(t,by=vz), data=DATA1)
	
	  #kvpX
      out.kvpX = gam(yvec~ s(t, by=vx)+s(t,by=vy)+s(t,by=vz)+px+py+pz, data=DATA1)
	
	  #kvsX
      out.kvsX = gam(yvec~ s(t, by=vx)+s(t,by=vy)+s(t,by=vz)+speed, data=DATA1)
		
	  #kvpsX
      out.kvpsX = gam(yvec~ s(t, by=vx)+s(t,by=vy)+s(t,by=vz)+px+py+pz+speed, data=DATA1)

      temp.sse = c(sum((predict(out.kd, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdp, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kds, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdps, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kv, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvp, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvs, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvps, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdpX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdsX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kdpsX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvpX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvsX, DATA2)-DATA2$yvec)^2),
        sum((predict(out.kvpsX, DATA2)-DATA2$yvec)^2)
        )
      cv.sse.mat[iiii,]=temp.sse
    }
	SSE.MAT.ALL[[j]]=cv.sse.mat
  }

  #The returned value
SSE.MAT.ALL
}


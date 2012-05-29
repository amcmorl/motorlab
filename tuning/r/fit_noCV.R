## Provides function:  fit_noCV
## Given simulated data, fit different models and report SSE (sum of squared error)
##    and AIC of each model
## zhanwu Liu Jan 27, 2010

## Input: a simulated or real dataset (assume object name is "simdata")
## The format is: 
##      simdata$firing.rate  The matrix of firing rate, each column corresponds to the firing
##                           rate simulated from one model
##      simdata$Xmat         The matrix contains simulated kinematics information
##                           columns: 1 time, 2-4 direction, 5-7 position, 8 speed, 
##                                    9-11 velocity, 12 target/trial id
##                                    for fitting only position should be used

## Returns: One object contains two matrices, let the object be "fitres", then
##     fitres$sse   The sum of squared error, each row for each set of firing rate input, 
##                  and columns for fit using different models
##     fitres$aic   The AIC score, arranged in the same way as fitres$sse
##     fitres$edf   The effective degrees of freedom of the fit in each model

fit_noCV = function (moddata, spl.k=10) {
  #moddata$Xmat[,c(2:4,8:11)] = 0
  #print('killing unreliable data')
  require(mgcv)

  ## how many sets of firing rate from same set of 
  ## kinematics information 
  ## And initiate the matrix used to store results
  n.dataset = dim(moddata$firing.rate)[2]
  mat.sse = matrix(0,n.dataset,16)
  mat.edf = matrix(0,n.dataset,16)
  
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
	
	colnames(Xmat) = c("t","dx","dy","dz","px","py","pz","speed","vx","vy","vz","tid")
	DATA1 = as.data.frame(cbind(yvec,Xmat))
	
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
	out.kdX = gam(yvec~ s(t,k=spl.k,by=dx) + s(t,k=spl.k,by=dy) + s(t,k=spl.k,by=dz),
      data=DATA1)
	
	#kdpX
	out.kdpX = gam(yvec~ s(t,k=spl.k,by=dx)+s(t,k=spl.k,by=dy)+s(t,k=spl.k,by=dz)
      + px + py + pz, data=DATA1)
	
	#kdsX
	out.kdsX = gam(yvec~ s(t,k=spl.k,by=dx)+s(t,k=spl.k,by=dy)+s(t,k=spl.k,by=dz)
      + speed, data=DATA1)
		
	#kdpsX
	out.kdpsX = gam(yvec~ s(t,k=spl.k,by=dx)+s(t,k=spl.k,by=dy)+s(t,k=spl.k,by=dz)
      + px + py + pz + speed, data=DATA1)

    
	#kvX
	out.kvX = gam(yvec~ s(t,k=spl.k,by=vx)+s(t,k=spl.k,by=vy)+s(t,k=spl.k,by=vz),
      data=DATA1)
	
	#kvpX
	out.kvpX = gam(yvec~ s(t,k=spl.k,by=vx)+s(t,k=spl.k,by=vy)+s(t,k=spl.k,by=vz)
      + px + py + pz, data=DATA1)
	
	#kvsX
	out.kvsX = gam(yvec~ s(t,k=spl.k,by=vx)+s(t,k=spl.k,by=vy)+s(t,k=spl.k,by=vz)
      + speed, data=DATA1)
		
	#kvpsX
	out.kvpsX = gam(yvec~ s(t,k=spl.k,by=vx)+s(t,k=spl.k,by=vy)+s(t,k=spl.k,by=vz)
      + px + py + pz + speed, data=DATA1)

	temp.sse = c(sum(out.kd$residuals^2),
				sum(out.kdp$residuals^2),
				sum(out.kds$residuals^2),
				sum(out.kdps$residuals^2),
				sum(out.kv$residuals^2),
				sum(out.kvp$residuals^2),
				sum(out.kvs$residuals^2),
				sum(out.kvps$residuals^2),
				sum(out.kdX$residuals^2),
				sum(out.kdpX$residuals^2),
				sum(out.kdsX$residuals^2),
				sum(out.kdpsX$residuals^2),
				sum(out.kvX$residuals^2),
				sum(out.kvpX$residuals^2),
				sum(out.kvsX$residuals^2),
				sum(out.kvpsX$residuals^2)
				)
    
 	temp.edf = c(sum(out.kd$edf),
				sum(out.kdp$edf),
				sum(out.kds$edf),
				sum(out.kdps$edf),
				sum(out.kv$edf),
				sum(out.kvp$edf),
				sum(out.kvs$edf),
				sum(out.kvps$edf),
				sum(out.kdX$edf),
				sum(out.kdpX$edf),
				sum(out.kdsX$edf),
				sum(out.kdpsX$edf),
				sum(out.kvX$edf),
				sum(out.kvpX$edf),
				sum(out.kvsX$edf),
				sum(out.kvpsX$edf)
				)
    
	mat.sse[j,] = temp.sse	
	mat.edf[j,] = temp.edf
	}
	#The returned value
	list(sse=mat.sse, edf=mat.edf)
}


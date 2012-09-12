## Provides function:  simu.8d
## simulate trials for 8-direction center out movement with gaussian speed profile
## zhanwu Liu Jan 25, 2010

## Input:
##   sd.factor is the factor used to add noise into the simulated firing rate
##             firing.rate ~ N(mean.firing.rate, sd = sd.factor*sqrt(mean.firing.rate))
##             default value 0, e.g. noiseless 

## Returns:
##   If the function was called using this:  simdata = simu.8d(sd.factor=0.001), then 
##   can get the following output:
##      simdata$firing.rate  The matrix of firing rate, each column corresponds to the firing
##                           rate simulated from one model
##      simdata$Xmat         The matrix contains simulated kinematics information
##                           columns: 1 time, 2-4 direction, 5-7 position, 8 speed, 
##                                    9-11 velocity, 12 target/trial id
##                                    for fitting only position should be used


source("common_functions.R")
simu.8d = function(sd.factor=0) {

	#how many data points on each trial: t.length+1
	#time is normalized to [0,1]
	t.length = 200
	time = (0:t.length)/t.length
	
	#speed profile is in Gaussian shape, max at time=0.5
	#sd = 0.12 so that about 93 points will be left if take 
	#cut off speed value at 15% of maximum
	#speed is also normalized
	speed.profile = dnorm(time, 0.5, 0.12)	
	speed = speed.profile/max(speed.profile)

	# The matrix for how each component of PD changes over time
	pd.changing = matrix(0, t.length+1, 3)
	
	for ( i in 1:(t.length+1)) {
			pd.changing[,1]=-time^2+1
			pd.changing[,2]=-(time-1)^2+1
			pd.changing[,3]=1
	}
	
	# The 8 directions to be simulated
	dir.mat = matrix(c(-1,-1,-1,
						-1,-1,1,
						-1,1,-1,
						-1,1,1,
						1,-1,-1,
						1,-1,1,
						1,1,-1,
						1,1,1),ncol=3,byrow=TRUE)
						
	# in straight line movement, distance traveled used for position
	distance.traveled = cumsum(speed)
	position.list = list(NULL)
	Xmat=numeric(0)
	for ( i in 1:8) {
		position.temp=(as.matrix(distance.traveled) %*% dir.mat[i,]  )
		velocity.temp = (as.matrix(speed) %*% dir.mat[i,]  )
		Xmat.temp =  cbind(time,matrix(rep(dir.mat[i,],t.length+1),byrow=TRUE,ncol=3), position.temp, speed, velocity.temp, i)
		Xmat=rbind(Xmat, Xmat.temp)
	}
	
	#The parameters for simulation -- need to adjust so that they will have similar contribution to firing rate
	k = 1
	para.d = c(0.1, 0.2, 0.3)
	para.p = c(0.3,0.4,0.5)/100
	para.s = 1
	para.v = c(0.4,0.5,0.6)
	
	#The matrix that will hold the simulated firing rate, each column for one model
	firing.rate = matrix(0, (t.length+1)*8, 16)

	## loop through different models	
	#kd 
	temp.para = c(0, para.d * 1, para.p*0, para.s * 0, para.v * 0, 0)
	firing.rate[,1] = k+Xmat %*% temp.para
	
	#kdp 
	temp.para = c(0, para.d * 1, para.p*1, para.s * 0, para.v * 0, 0)
	firing.rate[,2] = k+Xmat %*% temp.para
	#kds 
	temp.para = c(0, para.d * 1, para.p*0, para.s * 1, para.v * 0, 0)
	firing.rate[,3] = k+Xmat %*% temp.para
	#kdps 
	temp.para = c(0, para.d * 1, para.p*1, para.s * 1, para.v * 0, 0)
	firing.rate[,4] = k+Xmat %*% temp.para
	#kv
	temp.para = c(0, para.d * 0, para.p*0, para.s * 0, para.v * 1, 0)
	firing.rate[,5] = k+Xmat %*% temp.para
	#kvp 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 0, para.v * 1, 0)
	firing.rate[,6] = k+Xmat %*% temp.para
	#kvs 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 1, para.v * 1, 0)
	firing.rate[,7] = k+Xmat %*% temp.para
	#kvps 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 1, para.v * 1, 0)
	firing.rate[,8] = k+Xmat %*% temp.para
	
	#kdX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 0, para.v * 0, 0)
	firing.rate[,9] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,9] = firing.rate[i,9] + t(Xmat[i,c(2:4)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	
	#kdpX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 0, para.v * 0, 0)
	firing.rate[,10] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,10] = firing.rate[i,10] + t(Xmat[i,c(2:4)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	#kdsX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 1, para.v * 0, 0)
	firing.rate[,11] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,11] = firing.rate[i,11] + t(Xmat[i,c(2:4)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}

	#kdpsX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 1, para.v * 0, 0)
	firing.rate[,12] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,12] = firing.rate[i,12] + t(Xmat[i,c(2:4)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	
	#kvX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 0, para.v * 0, 0)
	firing.rate[,13] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,13] = firing.rate[i,13] + t(Xmat[i,c(9:11)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	
	#kvpX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 0, para.v * 0, 0)
	firing.rate[,14] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,14] = firing.rate[i,14] + t(Xmat[i,c(9:11)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	#kvsX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 1, para.v * 0, 0)
	firing.rate[,15] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,15] = firing.rate[i,15] + t(Xmat[i,c(9:11)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}

	#kvpsX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 1, para.v * 0, 0)
	firing.rate[,16] = k+Xmat %*% temp.para
	for ( i in 1:((t.length+1)*8)){
		firing.rate[i,16] = firing.rate[i,16] + t(Xmat[i,c(9:11)]) %*% pd.changing[(i-1) %%(t.length+1)+1,] 
	}
	
	##To add noise here
	# First make sure the firing rate is non-negative, because variance depends on it
	 if (min(firing.rate)<0) firing.rate=firing.rate-min(firing.rate)
	for ( j in 1:(dim(firing.rate)[2])) {
		firing.rate[,j] = rnorm(dim(firing.rate)[1], 0, sd=sd.factor * sqrt(firing.rate[,j]))+firing.rate[,j]
	}	
	
	## The last object is returned as output
	list(firing.rate = firing.rate, Xmat=Xmat)
}
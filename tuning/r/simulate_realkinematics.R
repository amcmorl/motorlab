## Provides function:  simu.realpos
## simulate trials from real kinematics data
## zhanwu Liu Jan 27, 2010

## Input:
##   sd.factor is the factor used to add noise into the simulated firing rate
##             firing.rate ~ N(mean.firing.rate, sd = sd.factor*sqrt(mean.firing.rate))
##             default value 0, e.g. noiseless 
##   traj.file is the file contains one R object named "pos.mat", which is a matrix
##             with four (4) columns.  The first three columns are the x, y, and z 
##             positions, and the fourth column is an indicator (integer) for trial 
##             so that the input can be split into trials

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
simu.realpos = function(sd.factor=0, traj.file = "pos.mat.RData") {
	load(traj.file) #real trajectories in pos.mat
	
		
	## Extract the needed information from pos.mat
	## calculate velocity, direction, speed
	Xmat = numeric(0)
	pd.changing = numeric(0)
	for ( i in unique(pos.mat[,4])) {
		ind.temp = which(pos.mat[,4]==i)
		#remove the first point from position so that the length would be same for 
		# position and velocity
		position.temp = pos.mat[ind.temp[-1],c(1:3)] 
		velocity.temp = matrix(0, length(ind.temp)-1, 3)
		direction.temp = matrix(0,length(ind.temp)-1,3)
		speed.temp = rep(0, length(ind.temp)-1)
		#normalize the time to be 0 to 1
		time.temp = c(2:length(ind.temp))/length(ind.temp)
		
		for ( j in 2:length(ind.temp)) {
			velocity.temp[j-1,] = pos.mat[ind.temp[j],c(1:3)]-pos.mat[ind.temp[j-1],c(1:3)]
			speed.temp[j-1]=vecnorm(velocity.temp[j-1,])
			direction.temp[j-1,] = velocity.temp[j-1,]/speed.temp[j-1]
		}
		Xmat.temp = cbind(time.temp, direction.temp, position.temp, speed.temp, velocity.temp, i)
		Xmat = rbind(Xmat, Xmat.temp)
		
		# Build the matrix for changing PD
		pd.changing.temp = matrix(0,length(ind.temp)-1, 3)
		pd.changing.temp[,1] = time.temp^2
		pd.changing.temp[,2] = (1-time.temp)^2
		pd.changing.temp[,3] = 1
		
		pd.changing = rbind(pd.changing, pd.changing.temp)
	}
	
	
	
	#The parameters for simulation -- need to adjust so that they will have similar contribution to firing rate
	k = 1
	para.d = c(0.1, 0.2, 0.3)
	para.p = c(0.3,0.4,0.5)
	para.s = 1
	para.v = c(0.4,0.5,0.6)*1000 # 1000 is used to adjust contribution
	                             # Also 1000 is used in the kv(??)X cases for the last four models
	
	
	## different models
	firing.rate = matrix(0, dim(Xmat)[1], 16)
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
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,9] = firing.rate[i,9] + t(Xmat[i,c(2:4)]) %*% pd.changing[i,] 
	}
	
	
	#kdpX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 0, para.v * 0, 0)
	firing.rate[,10] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,10] = firing.rate[i,10] + t(Xmat[i,c(2:4)]) %*% pd.changing[i,] 
	}
	
	#kdsX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 1, para.v * 0, 0)
	firing.rate[,11] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,11] = firing.rate[i,11] + t(Xmat[i,c(2:4)]) %*% pd.changing[i,] 
	}

	#kdpsX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 1, para.v * 0, 0)
	firing.rate[,12] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,12] = firing.rate[i,12] + t(Xmat[i,c(2:4)]) %*% pd.changing[i,] 
	}
	
	
	#kvX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 0, para.v * 0, 0)
	firing.rate[,13] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,13] = firing.rate[i,13] + t(Xmat[i,c(9:11)]) %*% pd.changing[i,] * 1000
	}
	
	
	#kvpX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 0, para.v * 0, 0)
	firing.rate[,14] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,14] = firing.rate[i,14] + t(Xmat[i,c(9:11)]) %*% pd.changing[i,] *1000
	}
	
	#kvsX 
	temp.para = c(0, para.d * 0, para.p*0, para.s * 1, para.v * 0, 0)
	firing.rate[,15] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,15] = firing.rate[i,15] +t(Xmat[i,c(9:11)]) %*% pd.changing[i,] *1000
	}

	#kvpsX 
	temp.para = c(0, para.d * 0, para.p*1, para.s * 1, para.v * 0, 0)
	firing.rate[,16] = k+Xmat %*% temp.para
	for ( i in 1:(length(pd.changing[,1]))){
		firing.rate[i,16] = firing.rate[i,16] + t(Xmat[i,c(9:11)]) %*% pd.changing[i,] *1000
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


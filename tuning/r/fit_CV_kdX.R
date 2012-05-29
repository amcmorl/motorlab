## Provides function:  fit_CV
## Given simulated data, randomly take half data, fit and predict in another half
##     repeat n.cv times and report the MSE from each random draw
## zhanwu Liu Jan 28, 2010
## added to by Angus McMorland, 20 Feb, 2010

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
##     e.g. each row is from one random draw, calculate the mse from fitting all 16 models

r_dir = '/home/amcmorl/files/pitt/tuning/code/r/'
source(paste(r_dir, 'common_functions.R', sep=""))
source(paste(r_dir, 'get_PD.R', sep=""))

fit_CV_kdX = function (moddata, n.cv=10, spl.k=10) {
  moddata$Xmat[,c(2:4,8:11)] = 0
  require(mgcv)
  ## The value for returning
  PREDICT.ALL   = list(NULL)
  ACTUAL.ALL    = list(NULL)
  TRIALS.ALL    = list(NULL)
  COEFS.ALL     = list(NULL)
  COEFNAMES.ALL = list(NULL)
  
  ## how many sets of firing rate from same set of 
  ## kinematics information 
  ## And initiate the matrix used to store results
  n.dataset = dim(moddata$firing.rate)[2]
  
  for (j in 1:n.dataset) {
    # Assign values from input to facilitate processing
	yvec = moddata$firing.rate[,j]
	Xmat.temp = moddata$Xmat
    
    # remove all NaNs up-front
    nas = is.na(yvec)
    yvec = yvec[!nas]
    Xmat.temp = Xmat.temp[!nas,]
    
	# Remove the first and last time point for each trial
	ind.remove = which(duplicated(Xmat.temp[,12])==FALSE)
	ind.remove = sort(c(ind.remove[-1]-1, ind.remove, length(Xmat.temp[,12])))
	yvec = yvec[-ind.remove]

	#reconstruct speed in the matrix
	trials.id  = unique(Xmat.temp[,12])
    n.trials = length(trials.id)
    n.train_trials = ceiling(n.trials / 2)
    n.test_trials = n.trials - n.train_trials
    
	Xmat = numeric(0)
	for (i in 1:n.trials) {
      ind.trial = which(Xmat.temp[,12]==trials.id[i])
      Xmat.trial = Xmat.temp[ind.trial,]
      pos.trial = Xmat.trial[,c(5:7)]
		
      vel.trial = matrix(0,length(pos.trial[,1])-1, 3)
      vel.trial[,1] = diff(pos.trial[,1])
      vel.trial[,2] = diff(pos.trial[,2])
      vel.trial[,3] = diff(pos.trial[,3])
		
      Xmat.trial = Xmat.trial[-1,] #remove first point
      Xmat.trial[,c(9:11)]=vel.trial

      speed.trial = apply(vel.trial, 1, vecnorm)
      Xmat.trial[,8]=speed.trial
     
      #Calculate direction 
      for ( ii in 1:length(vel.trial[,1])) {
        Xmat.trial[ii,c(2:4)] = vel.trial[ii,]/speed.trial[ii]
      }
		
      #Remove the last point
      Xmat.trial = Xmat.trial[-length(Xmat.trial[,1]),]
      n.pts_per_trial = length(Xmat.trial[,1])
		
      Xmat=rbind(Xmat, Xmat.trial)
	}

    #print(Xmat.trial)
    
    # the matrix to hold the cross-validation mse
    n.testpts = n.pts_per_trial * n.trials - n.pts_per_trial * n.train_trials
    predict.arr = array(0, c(n.cv, 16, n.testpts))
    testdata.arr = array(0, c(n.cv, n.testpts))
    test_trials.arr = array(0, c(n.cv, n.test_trials))
    coefs.arr = array(0, c(n.cv, 16))
    coefnames.arr = array(0, c(n.cv, 16))
    mode(coefs.arr) = 'list'
    mode(coefnames.arr) = 'list'
    
	colnames(Xmat) = c("t","dx","dy","dz","px","py","pz",
              "speed","vx","vy","vz","tid")
	ALLDATA = as.data.frame(cbind(yvec,Xmat))

	for ( i.cv  in 1:n.cv) {
      train_trials = sort(sample(trials.id, n.train_trials))
      
      ## +1 because trials.id starts at 0, and indexing in R starts at 1. Boo!
      trials = ALLDATA$tid
      in.train = trials %in% train_trials
      ind.train = which(in.train)
      ind.test = which(!in.train)
      test_trials = unique(ALLDATA$tid[ind.test])
      
      DATA1 = ALLDATA[ind.train,]
      DATA2 = ALLDATA[ind.test,]

      # save data for later inspection
      testdata.arr[i.cv,] = DATA2$yvec
      test_trials.arr[i.cv,] = test_trials
           
	  #kdX
      out.kdX = gam(log(yvec)~ s(t,by=dx) + s(t,by=dy) +
        s(t,by=dz), data=DATA1)

      smooth.pds = array(0, c(1, dim(DATA1)[1], 3))
      smooth.pds[1,,] = getPD(out.kdX, DATA1)
    }
    
    PREDICT.ALL[[j]]   = predict.arr
    ACTUAL.ALL[[j]]    = testdata.arr
    TRIALS.ALL[[j]]    = test_trials.arr
    COEFS.ALL[[j]]     = coefs.arr
    COEFNAMES.ALL[[j]] = coefnames.arr
  }

  #The returned value
  list(PREDICT.ALL, ACTUAL.ALL, TRIALS.ALL, COEFS.ALL, COEFNAMES.ALL, smooth.pds)
}

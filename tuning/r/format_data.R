## Inputs
## ------
## time    : flat array, length: n.bins
## pos     : rank 2,        dim: n.bins, 3
## trial   : flat array, length: n.bins
## sp_rate : rank 2,        dim: n.bins, n.datasets

## Returns
## -------
## simdata$firing.rate
##   The matrix of firing rate, each column corresponds to the firing
##   rate simulated from one model
## simdata$Xmat
##   The matrix contains simulated kinematics information
##   columns:
##     1 time
##     2-4 direction
##     5-7 position
##     8 speed
##     9-11 velocity
##     12 target/trial id
##   for fitting only position should be used

format_data = function (time, pos, trial, sp_rate) {
  n_data = dim(time)[1]
  dum3d = matrix(0, n_data, 3)
  dum1d = matrix(0, n_data, 1)
  dim(trial) = c(n_data, 1)
  Xmat = cbind(c(time), dum3d, pos, dum1d, dum3d,  trial)
  list('firing.rate' = sp_rate, 'Xmat' = Xmat)
}

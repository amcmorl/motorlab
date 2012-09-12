## To get the PD components
## gamout is the output for gam function
## DATA is the data input when fitting the gam model
getPD = function(gamout,DATA) {
  if (length(gamout$smooth) != 3) {
    stop("Check the length for gamout$smooth, should be 3")
  }
  n = dim(DATA)[1]
  pdmat = matrix(0,n,3)

  namelist = names(DATA)

  for ( i in 1:3) {
    by <- rep(1, n)
    
    ##get the column for time value
    term.col = which(namelist == gamout$smooth[[i]]$term)
    dat <- data.frame(x = DATA[, term.col], by = by)
    names(dat) <- c(gamout$smooth[[i]]$term, gamout$smooth[[i]]$by)
    
    X <- PredictMat(gamout$smooth[[i]], dat)
    first <- gamout$smooth[[i]]$first.para
    last <- gamout$smooth[[i]]$last.para
    p <- gamout$coefficients[first:last]
    offset <- attr(X, "offset")
    if (is.null(offset))
      fit <- X %*% p
    else fit <- X %*% p + offset
    
    pdmat[,i] = fit
  }
  pdmat
}

## Provides functions: vecnorm and Mat.normalization
## vecnorm: calculate the norm of a vector
## Mat.normalization: Given a matrix, return the matrix that each row is a 
##                    normalized row vector of the original matrix
## zhanwu Liu Jan 27, 2010


vecnorm <- function(x) {
	sqrt(sum(x^2))
}

Mat.normalization <- function(X){
	X.temp = X
	for ( i in 1:nrow(X)){
		X.temp[i,]=X[i,]/vecnorm(X[i,])
	}
	X.temp
}
r_dir = '/home/amcmorl/lib/python/motorlab/tuning/r'
source(paste(r_dir, 'common_functions.R', sep="/"))
source(paste(r_dir, 'get_PD.R', sep="/"))

fit_gam = function (data.train, data.test, model, predict, family) {

  cn = c("y", "t",
    "dx", "dy", "dz",
    "px", "py", "pz",
    "vx", "vy", "vz", "sp")
  colnames(data.train) = cn
  colnames(data.test)  = cn
  
  frame.train = as.data.frame(data.train)
  frame.test  = as.data.frame(data.test)
  
  #print(frame.train)
  #print(frame.test)

  if (model == "kd") {
    formula = y~dx + dy + dz
    
  } else if (model == "kdp") {
    formula = y~dx + dy + dz + px + py + pz
    
  } else if (model == "kds") {
    formula = y~dx + dy + dz + sp
    
  } else if (model == "kdps") {
    formula = y~dx + dy + dz + px + py + pz + sp
    
  } else if (model == "kv") {
    formula = y~vx + vy + vz
    
  } else if (model == "kvp") {
    formula = y~vx + vy + vz + px + py + pz
    
  } else if (model == "kvs") {
    formula = y~vx + vy + vz + sp
    
  } else if (model == "kvps") {
    formula = y~vx + vy + vz + px + py + pz + sp
    
  } else if (model == "kdX") {
    formula = y~s(t,by=dx) + s(t,by=dy) + s(t,by=dz)
    
  } else if (model == "kdpX") {
    formula = y~s(t,by=dx) + s(t,by=dy) + s(t,by=dz) + px + py + pz
    
  } else if (model == "kdsX") {
    formula = y~s(t,by=dx) + s(t,by=dy) + s(t,by=dz) + sp
    
  } else if (model == "kdpsX") {
    formula = y~s(t,by=dx) + s(t,by=dy) + s(t,by=dz) + px + py + pz + sp
    
  } else if (model == "kvX") {
    formula = y~s(t,by=vx) + s(t,by=vy) + s(t,by=vz)
    
  } else if (model == "kvpX") {
    formula = y~s(t,by=vx) + s(t,by=vy) + s(t,by=vz) + px + py + pz
    
  } else if (model == "kvsX") {
    formula = y~s(t,by=vx) + s(t,by=vy) + s(t,by=vz) + sp
    
  } else if (model == "kvpsX") {
    formula = y~s(t,by=vx) + s(t,by=vy) + s(t,by=vz) + px + py + pz + sp

  } else if (model == "kqX") {
    formula = y~dx + dy + dz + s(t,by=dx) + s(t,by=dy) + s(t,by=dz)
    
  } else if (model == 'null') {
    formula = y~1
    
  } else {
    print("Model not known")
    return()
  }

  # default to poisson if left blank
  if (family == "") {
    family = "poisson"
  }
  out = gam(formula, data=frame.train, family=family)
  logl = logLik.gam(out)

  coef = out$coefficients
  coefname = names(out$coefficients)
  
  if (length(grep("X", model)) > 0) {
    ## model has an 'X' in it
    smooth = getPD(out, frame.train)
  } else {
    smooth = 0
  }

  # don't predict for null model
  if ((predict == 1) & (model != 'null')) {
    # may want to not do this bit, because it takes time
    prediction = predict(out, frame.test, type='response')
  } else {
    prediction = rep(NA, nrow(frame.test))
  }
  list(coef, coefname, smooth, prediction, logl[1], attr(logl, 'df')[1])
  #list(coef, coefname, smooth, prediction)
}

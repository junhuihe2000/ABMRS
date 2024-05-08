# Manifold fitting with EBARS
# We refer readers to Principal manifold estimation for a detailed review.

reconstruction_map <- function(u, x) {
  p = ncol(x); d = ncol(u)
  maps_list = list()
  for(i in c(1:p)) {
    if(d==1) {bars_new = ebars(u, x[,i], intercept = T)}
    else if(d==2) {bars_new = binebars(u, x[,i], intercept_1 = T, intercept_2 = T)}
    bars_new$mcmc()
    maps_list[[i]] = bars_new
  }

  map = function(t) {return(sapply(maps_list, function(bars_class) {return(bars_class$predict(t))}))}
  return(map)
}

projection_func <- function(x, f, init_guess, lower_bound, upper_bound) {
  obj = function(t) {return(sum((x-f(rbind(t)))^2))}
  # res = nlm(obj, p = init_guess)
  res = optim(par=init_guess,fn=obj,lower=lower_bound,upper=upper_bound,method="L-BFGS-B")
  return(res$par)
}

#' Fit an 1 or 2 dimensional manifold with EBARS
#' @param x a numeric matrix, (n,p), each row indicates a point in R^p.
#' @param d int, the intrinsic dimension.
#' @param embedding_map function, with the usage `embedding_map(x,d)`,
#' return the embedding representation, a numeric matrix with dim(n,d).
#' @returns a list including:
#' \itemize{
#' \item{`map`, the estimated reconstruction map.}
#' \item{`u`, the manifold projection.}
#' \item{`fitted`, the manifold estimation on `u`.}
#' }
#' @export
#'
#' @examples
#' manifold=function(t){ return(c(cos(t),sin(t)))}
#' I=100
#' t=runif(I,min = 0,max = 1.5*pi)
#' X=manifold(t)
#' sd.noise=0.05
#' e1=rnorm(I,mean = 0, sd=sd.noise)
#' e2=rnorm(I,mean = 0, sd=sd.noise)
#' data.points=X+cbind(e1,e2)
#' embedding_map <- function(x, d) {
#'   u = vegan::isomap(stats::dist(x), ndim=d, k = 10)$points
#'   return(u)
#' }
#' res = manifold_fitting(data.points,1,embedding_map)
#' x_test = res$fitted
#' plot(data.points[,1], data.points[,2],
#' pch=20, col="grey",
#' main="Principal Manifold Estimation",
#' xlab=" ", ylab=" ")
#' lines(x_test[,1],x_test[,2],col="red",type = "l",lwd=3)
manifold_fitting <- function(x, d, embedding_map) {
  cat(paste0("Fit a ",d,"-dimensional manifold\n"))
  n = dim(x)[1]; p = dim(x)[2]
  u_initial = embedding_map(x, d)
  f = reconstruction_map(u_initial, x)

  if(FALSE) {
  lower = sapply(c(1:d),function(i) {return(min(u_initial[,i]))})
  upper = sapply(c(1:d),function(i) {return(max(u_initial[,i]))})
  # projection
  u = matrix(t(sapply(c(1:n), function(i) {return(projection_func(x[i,],f,u_initial[i,],lower,upper))})),ncol=d)
  }

  x_hat = f(u_initial)
  mse = mean(sqrt(rowSums((x-x_hat)^2)))
  cat(paste0("The reconstruction consistency loss is ",round(mse,5),".\n"))

  if(FALSE) {
  # iteration
  count = 1;
  ssd_ratio = 1
  cat(paste0("The ",count,"-th iteration: ","the consistency loss is ",round(ssd,5),".\n"))
  while(ssd_ratio>epsilon & count < max_iter) {
    count = count + 1
    f_new = embedding_map(u, x)
    lower = sapply(c(1:d),function(i) {return(min(u[,i]))})
    upper = sapply(c(1:d),function(i) {return(max(u[,i]))})
    u_new = matrix(t(sapply(c(1:n), function(i) {return(projection_func(x[i,],f_new,u[i,],lower,upper))})),ncol=d)
    ssd_new = sum((x-f_new(u_new))^2)

    ssd_ratio = (ssd-ssd_new)/ssd
    if(ssd > ssd_new) {
      f = f_new; u = u_new; ssd = ssd_new
    }
    cat(paste0("The ",count,"-th iteration: ","the consistency loss is ",round(ssd,5),".\n"))
  }
  }
  return(list(map=f, u=u_initial, fitted=x_hat))
}

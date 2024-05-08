# wrappers for exposed Rcpp classes

#' create an instance of EBARS
#' @description
#' see [ClassEBARS] for more details about class EBARS
#'
#'
#' @param x a numeric vector, indicates the predictor values.
#' @param y a numeric vector, indicates the response values, the same length as x.
#' @param gamma double, the exponent in extended BIC, default value is `1.0`.
#' It should be contained in `[0,1]` (Nevertheless, outside is valid).
#' @param c double, the constant affects the proposal distribution in MCMC,
#' default value is `0.3`. It has to be contained in `[0,1]`.
#' @param times double, decides the number of all potential knots if `n<0`.
#' In this case, the number is `length(x) * times`, default value is `3.0`.
#' @param k int, the number of knots is fixed if `k>0`. Otherwise `k` will
#' be estimated in the algorithm, default value is `-1`.
#' @param n int, the number of all potential knots, default value is `-1`.
#' @param degree int, the degree of polynomial spline, default value is `3`.
#' @param intercept bool, whether the intercept is included in the basis,
#' default value is `TRUE`.
#' @returns an exposed R class of cpp class called EBARS.
#' @export
#' @examples
#' library(EBARS)
#' library(splines)
#' set.seed(1234)
#' knot = c(0.4,0.4,0.4,0.4,0.7)
#' beta = matrix(c(2,-5,5,2,-3,-1,2),ncol=1)
#' m = 200
#'
#' # generate train data set
#' x = c(0,sort(runif(m, 0, 1))[c(-1,-m)],1)
#' B = ns(x,knots=knot,intercept=TRUE,Boundary.knots=c(0,1))
#' y = B%*%beta
#' y_h = y + rnorm(m,0,0.05)
#'
#' # run EBARS
#' a = ebars(x,y_h,c=0.3,times = 2)
#' a$mcmc()
#' print(a$knots())
#'
#' # generate test data set
#' m_new = 50
#' x_new = runif(m_new,0,1)
#' y_new = predict(B,x_new)%*%beta
#' y_pred = a$predict(x_new)
#'
ebars <- function(x, y, gamma = 1.0, c = 0.3, times = 3,
                  k = -1, n = -1, degree = 3, intercept = TRUE) {
  new_ebars = EBARS$new(x,y,c(gamma,c,times),c(k,n),list("degree"=degree,"intercept"=intercept))
  assign('mcmc',
         function(burns = 500, steps = 500) {
           new_ebars$rjmcmc(burns,steps)
         },
         envir = new_ebars)
  return(new_ebars)
}



#' create an instance of BinEBARS
#' @description
#' see [ClassBinEBARS] for more details about class BinEBARS
#'
#'
#' @param x a numeric matrix, (m,2), each row indicates a predictor value (x_1,x_2).
#' @param y a numeric vector, indicates the response values,
#' the same length as x's rows.
#' @param gamma double, the exponent in extended BIC, default value is `1.0`.
#' It should be contained in `[0,1]` (Nevertheless, outside is valid).
#' @param c double, the constant affects the proposal distribution in MCMC,
#' default value is `0.3`. It has to be contained in `[0,1]`.
#' @param times_1 double, decides the number of all potential knots in x_1 if `n_1<0`.
#' In this case, `n_1` is `length(x) * times`, default value is `3.0`.
#' @param times_2 double, similar to `times_1` but in x_2, default value is `3.0`.
#' @param k_1 int, the number of knots in x_1 is fixed if `k_1>0`. Otherwise `k_1` will
#' be estimated in the algorithm, default value is `-1`.
#' @param k_2 int, the number of knots in x_2 is fixed if `k_2>0`. Otherwise `k_2` will
#' be estimated in the algorithm, default value is `-1`.
#' @param n_1 int, the number of all potential knots in x_1, default value is `-1`.
#' @param n_2 int, similar to `n_1` but in x_2, default value is `-1`.
#' @param degree_1 int, the degree of polynomial spline in x_1, default value is `3`.
#' @param degree_2 int, the degree of polynomial spline in x_2, default value is `3`.
#' @param intercept_1 bool, whether the intercept is included in the basis in x_1,
#' default value is `TRUE`.
#' @param intercept_2 bool, whether the intercept is included in the basis in x_2,
#' default value is `TRUE`.
#' @returns an exposed R class of cpp class called BinEBARS.
#' @export
#' @examples
#' library(EBARS)
#' library(splines)
#' set.seed(1234)
#' #tensor spline
#' beta = matrix(rnorm(40,0,1),ncol=1)
#' fss <- function(x,y){
#'   xix = c(0.2,0.3)
#'   xiy = c(0.5,0.5,0.5,0.5,0.7)
#'   B = tensor_spline(cbind(x,y),xix,xiy)
#'   return(B%*%beta)
#' }
#' # parameters' configuration
#' m_train = 1000; m_test = 200
#' burns = 1000; steps = 1000
#' noise = 0.1
#' # generate train data set
#' x_1 = c(runif(m_train-2,0,1),0,1)
#' x_2 = c(runif(m_train-2,0,1),0,1)
#' y = fss(x_1,x_2)
#' y_h = y + rnorm(m_train,0,noise)
#' # generate test set
#' x_1_new = runif(m_test,0,1)
#' x_2_new = runif(m_test,0,1)
#' y_new = fss(x_1_new,x_2_new)
#' # run suface EBARS
#' time_start = Sys.time()
#' my_binebars = binebars(cbind(x_1,x_2), y_h)
#' my_binebars$mcmc(10,10)
#' time_end = Sys.time()
#' time_end - time_start
#' y_hat = my_binebars$predict(cbind(x_1_new,x_2_new))
#' sum((y_new-y_hat)^2)/m_test
#'
binebars <- function(x, y, gamma = 1.0, c = 0.3, times_1 = 3, times_2 = 3,
                     k_1 = -1, k_2 = -1, n_1 = -1, n_2 = -1,
                     degree_1 = 3, degree_2 = 3, intercept_1 = TRUE, intercept_2 = TRUE) {
  stopifnot("x must be an (m,2) matrix"=ncol(x)==2)
  new_binebars = BinEBARS$new(x,y,c(gamma,c,times_1,times_2),c(k_1,k_2,n_1,n_2),
                              list("degree_1"=degree_1,"degree_2"=degree_2,"intercept_1"=intercept_1,"intercept_2"=intercept_2))
  assign('mcmc',
         function(burns = 100, steps = 100) {
           new_binebars$rjmcmc(burns,steps)
         },
         envir = new_binebars)
  return(new_binebars)
}

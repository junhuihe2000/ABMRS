# wrappers for exposed Rcpp classes

#' create an instance of EBARS, see `ClassEBARS` for more details about EBARS
#'
#' @param x a numeric vector, indicates the predictor values.
#' @param y a numeric vector, indicates the response values, the same length as x.
#' @param gamma double, the exponent in extended BIC, default value is `1.0`.
#' It should be contained in `[0,1]` (Nevertheless, outside is valid).
#' @param c double, the constant affects the proposal distribution in MCMC,
#' default value is `0.3`. It has to be contained in `[0,1]`.
#' @param times int, decides the number of all potential knots if `n<0`.
#' In this case, the number is `length(x) * times`, default value is `2`.
#' @param n int, the number of all potential knots, default value is `-1`.
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
#' x = c(0,sort(runif(m_train, 0, 1))[c(-1,-m)],1)
#' B = ns(x,knots=knot,intercept=TRUE,Boundary.knots=c(0,1))
#' y = B%*%beta
#' y_h = y + rnorm(m_train,0,0.05)
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
ebars <- function(x, y, gamma = 1.0, c = 0.3, times = 2, n = -1) {
  new_ebars = EBARS$new(x,y,gamma,c,times,n)
  assign('mcmc',
         function(burns = 500, steps = 500, flush = FALSE, gap = 50) {
           new_ebars$rjmcmc(burns,steps,flush,gap)
         },
         envir = new_ebars)
  return(new_ebars)
}

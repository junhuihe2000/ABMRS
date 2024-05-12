
<!-- README.md is generated from README.Rmd. Please edit that file -->

# EBARS

<!-- badges: start -->
<!-- badges: end -->

The goal of EBARS is to implement the extended Bayesian adaptive
regression spline algorithm of the article ***Extended Bayesian
information criterion for multivariate spline knot inference***. In
spline regression, the number and location of knots influence the
performance and interpretability significantly. We propose a fully
Bayesian approach for knot inference in multivariate spline regression.
The knot inference is of interest in many problems, such as change point
detection. We can estimate the knot number and location simultaneously
and accurately by the proposed method in univariate or bivariate
splines.

We specify a prior on the knot number to take into account the
complexity of the model space and derive an analytic formula in the
normal model. In the non-normal cases, we utilize the extended Bayesian
information criterion (EBIC) to approximate the posterior density. The
samples are simulated in the space with differing dimensions via
reversible jump Markov chain Monte Carlo (RJMCMC). Experiments
demonstrate the splendid capability of the algorithm, especially in
function fitting with jumping discontinuity.

## Installation

You can install the development version of EBARS from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("junhuihe2000/EBARS")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(EBARS)
library(splines)
library(ggplot2)
library(ggpubr)

f1 = function(x) {
  knots = c(0.5)
  beta = c(1,-1,1)
  B = bs(x,knots=knots,degree=1,intercept=TRUE,Boundary.knots=c(0,1))
  y = B%*%beta
  return(y)
}

f2 = function(x) {
  knots = c(0.3,0.7)
  beta = c(2,-1,-2,-1)
  B = bs(x,knots=knots,degree=1,intercept=TRUE,Boundary.knots=c(0,1))
  y = B%*%beta
  return(y)
}

f3 = function(x) {
  knots = c(0.2,0.2,0.5,0.7)
  beta = c(0,-1,1,0,1,0)
  B = bs(x,knots=knots,degree=1,intercept=TRUE,Boundary.knots=c(0,1))
  y = B%*%beta
  return(y)
}

set.seed(1234)
m = 200
x_train = runif(m,0,1)
# case 1
y_train_1 = f1(x_train)
y_obs_1 = y_train_1 + rnorm(m,0,0.4)
p1 = ggplot() + geom_point(aes(x_train,y_obs_1),size=0.6,col="red") + geom_line(aes(x_train,y_train_1)) + theme(axis.title.x=element_blank(),axis.title.y=element_blank())
# case 2
y_train_2 = f2(x_train)
y_obs_2 = y_train_2 + rnorm(m,0,0.3)
p2 = ggplot() + geom_point(aes(x_train,y_obs_2),size=0.6,col="red") + geom_line(aes(x_train,y_train_2)) + theme(axis.title.x=element_blank(),axis.title.y=element_blank())
# case 3 
y_train_3 = f3(x_train)
y_obs_3 = y_train_3 + rnorm(m,0,0.4)
p3 = ggplot() + geom_point(aes(x_train,y_obs_3),size=0.6,col="red") + geom_line(aes(x_train,y_train_3)) + theme(axis.title.x=element_blank(),axis.title.y=element_blank())

# arrange in one page
annotate_figure(ggarrange(p1,p2,p3,ncol=3), top="Linear splines with one, two and four knots")
```

<img src="man/figures/README-data-1.png" width="100%" />

<div class="figure">

<embed src="inst/figures/knot_estimation.pdf" title="Linear splines with one, two and four knots." width="100%" type="application/pdf" />
<p class="caption">
Linear splines with one, two and four knots.
</p>

</div>

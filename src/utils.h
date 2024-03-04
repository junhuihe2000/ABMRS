#ifndef UTILS_H
#define UTILS_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>


// estimate beta and sigma in the spline regression by least square estimation
Rcpp::List spline_regression(const Eigen::VectorXd & x, const Eigen::VectorXd & y,
                             const Eigen::VectorXd & xi);


#endif

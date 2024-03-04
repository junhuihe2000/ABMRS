// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <algorithm>

#include "utils.h"

Rcpp::List spline_regression(const Eigen::VectorXd & x, const Eigen::VectorXd & y,
                             const Eigen::VectorXd & xi) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x),
                         Rcpp::Named("knots")=Rcpp::wrap(xi)));
  Eigen::VectorXd beta = (B.transpose()*B).ldlt().solve(B.transpose()*y);
  double sigma = (y-B*beta).norm() / std::sqrt(x.size());
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            Rcpp::Named("sigma")=sigma);
}



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
                         Rcpp::Named("knots")=Rcpp::wrap(xi),
                         Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::VectorXd beta = (B.transpose()*B).ldlt().solve(B.transpose()*y);
  double sigma = (y-B*beta).norm() / std::sqrt(x.size());
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            Rcpp::Named("sigma")=sigma);
}

Eigen::VectorXd spline_predict(const Eigen::VectorXd & x_new, const Eigen::VectorXd & xi,
                               const Eigen::VectorXd & beta) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x_new),
                                                   Rcpp::Named("knots")=Rcpp::wrap(xi),
                                                   Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::VectorXd y_new = B*beta;
  return y_new;
}


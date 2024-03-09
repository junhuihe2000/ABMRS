#ifndef UTILS_H
#define UTILS_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>


// estimate beta and sigma in the spline regression by least square estimation
Rcpp::List spline_regression(const Eigen::VectorXd & x, const Eigen::VectorXd & y,
                             const Eigen::VectorXd & xi,
                             int degree = 3, bool intercept = false);

// predict response values on x_new by spline regression
Eigen::VectorXd spline_predict(const Eigen::VectorXd & x_new, const Eigen::VectorXd & xi,
                               const Eigen::VectorXd & beta,
                               int degree = 3, bool intercept = false);

// estimate beta and sigma in the surface spline regression by least square estimation
Rcpp::List surface_spline_regression(const Eigen::MatrixXd & x,
                                     const Eigen::VectorXd & y,
                                     const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                                     int degree_1 = 3, int degree_2 = 3,
                                     bool intercept_1 = false, bool intercept_2 = false);

//' create a bivariate tensor product spline basis matrix
//' @param x a numeric matrix, (m,2), each row indicates a predictor value (x_1,x_2).
//' @param xi_1 a numeric vector, indicates `k_1` knots in x_1.
//' @param xi_2 a numeric vector, indicates `k_2` knots in x_2
//' @param degree_1 int, the degree of polynomial in x_1, default value is `3`.
//' @param degree_2 int, the degree of polynomial in x_2, default value is `3`.
//' @param intercept_1 bool, whether the intercept is included in the basis in x_1,
//' default value is `FALSE`.
//' @param intercept_2 bool, whether the intercept is included in the basis in x_2,
//' default value is `FALSE`.
//'
//' @returns a bivariate tensor product spline basis matrix, m rows and (k_1+3)*(k_2+3) cols.
//'
//' @export
// [[Rcpp::export(tensor_spline)]]
Eigen::MatrixXd tensor_spline(const Eigen::MatrixXd & x,
                              const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                              int degree_1 = 3, int degree_2 = 3,
                              bool intercept_1 = false, bool intercept_2 = false);

// predict response values on x_new by surface spline regression
Eigen::VectorXd surface_spline_predict(const Eigen::MatrixXd & x_new,
                                       const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                                       const Eigen::VectorXd & beta,
                                       int degree_1 = 3, int degree_2 = 3,
                                       bool intercept_1 = false, bool intercept_2 = false);

#endif

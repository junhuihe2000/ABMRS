// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <algorithm>

#include "utils.h"

Rcpp::List spline_regression(const Eigen::VectorXd & x, const Eigen::VectorXd & y,
                             const Eigen::VectorXd & xi, int degree, bool intercept) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x),
                         Rcpp::Named("knots")=Rcpp::wrap(xi),
                         Rcpp::Named("degree")=degree,
                         Rcpp::Named("intercept")=intercept,
                         Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::VectorXd beta = (B.transpose()*B).ldlt().solve(B.transpose()*y);
  double sigma = (y-B*beta).norm() / std::sqrt(x.size());
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            Rcpp::Named("sigma")=sigma);
}

Eigen::VectorXd spline_predict(const Eigen::VectorXd & x_new, const Eigen::VectorXd & xi,
                               const Eigen::VectorXd & beta,
                               int degree, bool intercept) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x_new),
                                                   Rcpp::Named("knots")=Rcpp::wrap(xi),
                                                   Rcpp::Named("degree")=degree,
                                                   Rcpp::Named("intercept")=intercept,
                                                   Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::VectorXd y_new = B*beta;
  return y_new;
}

Eigen::MatrixXd tensor_spline(const Eigen::MatrixXd & x,
                              const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                              int degree_1, int degree_2,
                              bool intercept_1, bool intercept_2) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B_1 = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x.col(0)),
                                                     Rcpp::Named("knots")=Rcpp::wrap(xi_1),
                                                     Rcpp::Named("degree")=degree_1,
                                                     Rcpp::Named("intercept")=intercept_1,
                                                     Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::MatrixXd B_2 = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x.col(1)),
                                                     Rcpp::Named("knots")=Rcpp::wrap(xi_2),
                                                     Rcpp::Named("degree")=degree_2,
                                                     Rcpp::Named("intercept")=intercept_2,
                                                     Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));

  Eigen::MatrixXd B(B_1.rows(), (B_1.cols()*B_2.cols()));
  for(int i=0;i<B.rows();i++) {
    B.row(i) = Eigen::kroneckerProduct(B_1.row(i), B_2.row(i));
  }

  return B;
}


Eigen::MatrixXd tri_tensor_spline(const Eigen::MatrixXd & x,
                              const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2, const Eigen::VectorXd & xi_3,
                              int degree_1, int degree_2, int degree_3,
                              bool intercept_1, bool intercept_2, bool intercept_3) {
  // generate B-spline design matrix
  Rcpp::Environment splines = Rcpp::Environment::namespace_env("splines");
  Rcpp::Function bs = splines["bs"];

  Eigen::MatrixXd B_1 = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x.col(0)),
                                                     Rcpp::Named("knots")=Rcpp::wrap(xi_1),
                                                     Rcpp::Named("degree")=degree_1,
                                                     Rcpp::Named("intercept")=intercept_1,
                                                     Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::MatrixXd B_2 = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x.col(1)),
                                                     Rcpp::Named("knots")=Rcpp::wrap(xi_2),
                                                     Rcpp::Named("degree")=degree_2,
                                                     Rcpp::Named("intercept")=intercept_2,
                                                     Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));
  Eigen::MatrixXd B_3 = Rcpp::as<Eigen::MatrixXd>(bs(Rcpp::Named("x")=Rcpp::wrap(x.col(2)),
                                                     Rcpp::Named("knots")=Rcpp::wrap(xi_3),
                                                     Rcpp::Named("degree")=degree_3,
                                                     Rcpp::Named("intercept")=intercept_3,
                                                     Rcpp::Named("Boundary.knots")=Rcpp::NumericVector::create(0.0,1.0)));

  Eigen::MatrixXd B(B_1.rows(), (B_1.cols()*B_2.cols()*B_3.cols()));

  for(int i=0;i<B.rows();i++) {
    B.row(i) = Eigen::kroneckerProduct(Eigen::MatrixXd(Eigen::kroneckerProduct(B_1.row(i), B_2.row(i))), B_3.row(i));
  }


  return B;
}



Rcpp::List cube_spline_regression(const Eigen::MatrixXd & x,
                                     const Eigen::VectorXd & y,
                                     const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2, const Eigen::VectorXd & xi_3,
                                     int degree_1, int degree_2, int degree_3,
                                     bool intercept_1, bool intercept_2, bool intercept_3) {
  Eigen::MatrixXd B = tri_tensor_spline(x, xi_1, xi_2, xi_3, degree_1, degree_2, degree_3, intercept_1, intercept_2, intercept_3);
  Eigen::VectorXd beta = (B.transpose()*B).ldlt().solve(B.transpose()*y);
  double sigma = (y-B*beta).norm() / std::sqrt(x.rows());
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            Rcpp::Named("sigma")=sigma);
}


Rcpp::List surface_spline_regression(const Eigen::MatrixXd & x,
                                     const Eigen::VectorXd & y,
                                     const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                                     int degree_1, int degree_2,
                                     bool intercept_1, bool intercept_2) {
  Eigen::MatrixXd B = tensor_spline(x, xi_1, xi_2, degree_1, degree_2, intercept_1, intercept_2);
  Eigen::VectorXd beta = (B.transpose()*B).ldlt().solve(B.transpose()*y);
  double sigma = (y-B*beta).norm() / std::sqrt(x.rows());
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            Rcpp::Named("sigma")=sigma);
}


Eigen::VectorXd surface_spline_predict(const Eigen::MatrixXd & x_new,
                                       const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2,
                                       const Eigen::VectorXd & beta,
                                       int degree_1, int degree_2,
                                       bool intercept_1, bool intercept_2) {
  Eigen::MatrixXd B = tensor_spline(x_new, xi_1, xi_2, degree_1, degree_2, intercept_1, intercept_2);
  Eigen::VectorXd y_new = B*beta;
  return y_new;
}


Eigen::VectorXd cube_spline_predict(const Eigen::MatrixXd & x_new,
                                       const Eigen::VectorXd & xi_1, const Eigen::VectorXd & xi_2, const Eigen::VectorXd & xi_3,
                                       const Eigen::VectorXd & beta,
                                       int degree_1, int degree_2, int degree_3,
                                       bool intercept_1, bool intercept_2, bool intercept_3) {
  Eigen::MatrixXd B = tri_tensor_spline(x_new, xi_1, xi_2, xi_3, degree_1, degree_2, degree_3, intercept_1, intercept_2, intercept_3);
  Eigen::VectorXd y_new = B*beta;
  return y_new;
}


#ifndef EBARS_H
#define EBARS_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "utils.h"

//' @name ClassEBARS
//' @title extended Bayesian adaptive regression spline
//' @description a class of [ebars()]
//'
//' @field new constructor, see `ebars`
//' @field rjmcmc reversible jump MCMC, a wrapper is `mcmc`
//' @field predict predict by spline regression with EBARS
//' @field knots return estimated knots
//' @field samples return posterior samples
class EBARS {
private:
  int k; // the number of underlying knots
  int n; // the number of all potential knots
  int m; // the number of training points
  int degree; // the degree of polynomials
  double gamma, c; // constants
  double xmin, xmax; // range of x
  bool fix_k; // whether k is given and fixed
  bool intercept; // whether an intercept is included in the basis

  Eigen::VectorXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd t;

  Eigen::VectorXd knots; // all knots
  Eigen::VectorXd xi; // selected knots
  Eigen::VectorXd remain_knots; // remaining knots
  Rcpp::List xis; // posterior samples of selected knots

  Eigen::VectorXd beta; // estimated beta by MLE
  double sigma; // estimated sigma by MLE


  // compute probabilities of the next movement type
  double _birth();
  double _death();
  double _relocate();

  void _knots(); // create equally-spaced knot locations
  void _initial(); // initialize

  bool _jump(); // one possible jump, may be rejected
  void _update(); // one step movement in RJMCMC

public:
  // initialize EBARS
  EBARS(const Eigen::VectorXd & _x, const Eigen::VectorXd & _y,
        Rcpp::NumericVector _para = Rcpp::NumericVector::create(1.0,0.3,2.0),
        Rcpp::IntegerVector _num = Rcpp::IntegerVector::create(-1,-1),
        Rcpp::List _spline = Rcpp::List::create(Rcpp::Named("degree") = 3,
                                                Rcpp::Named("intercept") = false));
  void rjmcmc(int burns = 500, int steps = 500); // reversible jump MCMC
  // predict response values on x_new
  Eigen::VectorXd predict(const Eigen::VectorXd & x_new);
  Eigen::VectorXd get_knots(); // return estimated knots
  Rcpp::List get_samples(); // return posterior samples
};

#endif

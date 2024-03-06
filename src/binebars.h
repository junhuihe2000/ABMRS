#ifndef BINEBARS_H
#define BINEBARS_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include "utils.h"


//' @name ClassBinEBARS
//' @title bivariate extended Bayesian adaptive regression spline
//' @description a class of [binebars()]
//'
//' @field new constructor, see `binebars`
//' @field rjmcmc reversible jump MCMC, a wrapper is `mcmc`
//' @field predict predict by surface spline regression with EBARS
//' @field knots return estimated knots
class BinEBARS {
private:
  int k_1, k_2; // the number of underlying knots in the direction x_1 and x_2
  int n_1, n_2; // the number of all potential knots in the direction x_1 and x_2
  int m; // the number of training points
  double gamma, c; // constants
  Eigen::RowVector2d xmin, xmax; // range of x

  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::MatrixXd t;

  Eigen::VectorXd knots_1, knots_2; // all knots
  Eigen::VectorXd xi_1, xi_2; // selected knots
  Eigen::VectorXd remain_knots_1, remain_knots_2; // remaining knots

  Eigen::VectorXd beta; // estimated beta by MLE
  double sigma; // estimated sigma by MLE


  // compute probabilities of the next movement type
  double _birth_1(); double _birth_2();
  double _death_1(); double _death_2();
  double _relocate_1(); double _relocate_2();

  void _knots(); // create equally-spaced knot locations
  void _initial(); // initialize

  bool _jump_1(); // one possible jump in the direction x_1, may be rejected
  bool _jump_2(); // one possible jump in the direction x_2, may be rejected
  void _update(); // one step movement in RJMCMC

public:
  // initialize BinEBARS
  BinEBARS(const Eigen::MatrixXd & _x, const Eigen::VectorXd & _y,
           double _gamma = 1.0, double _c = 0.3,
           Rcpp::NumericVector _times = Rcpp::NumericVector::create(1.0,1.0),
           Rcpp::IntegerVector _n = Rcpp::IntegerVector::create(-1,-1));
  void rjmcmc(int burns = 200, int steps = 200); // reversible jump MCMC
  // predict response values on x_new
  Eigen::VectorXd predict(const Eigen::MatrixXd & x_new);
  Rcpp::List get_knots(); // return estimated knots
};





#endif

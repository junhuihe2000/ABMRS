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
class EBARS {
private:
  int k; // the number of underlying knots
  int n; // the number of all potential knots
  int m; // the number of training points
  double gamma, c; // constants
  double xmin, xmax; // range of x

  Eigen::VectorXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd t;

  Eigen::VectorXd knots; // all knots
  Eigen::VectorXd xi; // selected knots
  Eigen::VectorXd remain_knots; // remaining knots

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
        double _gamma = 1.0, double _c = 0.3, double _times = 2.0, int _n = -1);
  void rjmcmc(int burns = 500, int steps = 500, bool flush = false, int gap = 50); // reversible jump MCMC
  // predict response values on x_new
  Eigen::VectorXd predict(const Eigen::VectorXd & x_new);
  Eigen::VectorXd get_knots(); // return estimated knots
};

#endif

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <algorithm>

#include "ebars.h"

double EBARS::_birth() {
  double p = c * std::min(1.0, std::pow((n-k)/(k+1.0), 1.0-gamma));
  return p;
}

double EBARS::_death() {
  double p = c * std::min(1.0, std::pow(k/(n-k+1.0), 1.0-gamma));
  return p;
}

double EBARS::_relocate() {
  double p = 1.0 - _birth() - _death();
  return p;
}

void EBARS::_knots() {
  knots = Eigen::VectorXd::Zero(n);
  double step = 1.0/(n+1);
  knots(0) = step;
  for(int i=1;i<n;i++) {
    knots(i) = knots(i-1) + step;
  }
}

EBARS::EBARS(const Eigen::VectorXd & _x, const Eigen::VectorXd & _y,
             Rcpp::NumericVector _para, Rcpp::IntegerVector _num, Rcpp::List _spline) {
  x = _x; y = _y;
  gamma = _para(0); c = _para(1);
  m = _y.size();
  double _times = _para(2);
  // compute the number of knots
  int _k = _num(0);
  fix_k = (_k>0) ? true : false;
  k = (_k>0) ? _k : 1;

  int _n = _num(1);
  n = (_n>0) ? _n : int(m*_times);

  degree = int(_spline["degree"]);
  intercept = _spline["intercept"];

  // transform x to t
  xmin = x.minCoeff(); xmax = x.maxCoeff();
  t = (x.array()-xmin)/(xmax-xmin);

  _knots();
  _initial();
  // maximum likelihood estimation
  Rcpp::List pars = spline_regression(t, y, xi, degree, intercept);
  beta = Rcpp::as<Eigen::VectorXd>(pars["beta"]);
  sigma = pars["sigma"];
}

void EBARS::_initial() {
  Eigen::VectorXi idx = Rcpp::as<Eigen::VectorXi>(Rcpp::sample(n, k)).array()-1;
  Eigen::VectorXi idx_rem = Eigen::VectorXi::LinSpaced(n, 0, n-1);
  xi = Eigen::VectorXd::Zero(k);
  for(int i=0;i<k;i++) {
    xi(i) = knots(idx(i));
    idx_rem(idx(i)) = -1; // remark selected knots as -1
  }

  remain_knots = Eigen::VectorXd::Zero(n-k);
  int j = 0;
  for(int i=0;i<n;i++) {
    if(idx_rem(i)>=0) {
      remain_knots(j++) = knots(idx_rem(i));
    }
  }
}

bool EBARS::_jump() {
  double type = Rcpp::runif(1, 0.0, 1.0)[0];
  double birth = _birth();
  double death = _death();
  if(k==1) {death = 0.0;}
  if(fix_k) {birth = 0.0; death = 0.0;}

  Eigen::VectorXd xi_new, remain_new;
  int k_new;

  // birth scheme
  if(type < birth) {
    k_new = k + 1;
    int idx = Rcpp::sample(n-k,1)[0] - 1;
    xi_new = Eigen::VectorXd::Zero(k_new);
    remain_new = Eigen::VectorXd::Zero(n-k_new);
    xi_new.head(k) = xi; xi_new(k) = remain_knots(idx);
    std::sort(xi_new.data(),xi_new.data()+xi_new.size());
    remain_new.head(idx) = remain_knots.head(idx);
    remain_new.tail(n-k_new-idx) = remain_knots.tail(n-k_new-idx);
  }

  // death scheme
  else if(type < (birth+death)) {
    k_new = k - 1;
    int idx = Rcpp::sample(k,1)[0] - 1;
    xi_new = Eigen::VectorXd::Zero(k_new);
    remain_new = Eigen::VectorXd::Zero(n-k_new);
    xi_new.head(idx) = xi.head(idx);
    xi_new.tail(k_new-idx) = xi.tail(k_new-idx);
    remain_new.head(n-k) = remain_knots; remain_new(n-k) = xi(idx);
  }

  // relocate scheme
  else {
    k_new = k;
    int idx_0 = Rcpp::sample(k,1)[0] - 1;
    int idx_1 = Rcpp::sample(n-k,1)[0] - 1;
    xi_new = xi; remain_new = remain_knots;
    xi_new(idx_0) = remain_knots(idx_1);
    remain_new(idx_1) = xi(idx_0);
    std::sort(xi_new.data(),xi_new.data()+xi_new.size());
  }

  // compute MLE
  Rcpp::List pars_new = spline_regression(t,y,xi_new,degree,intercept);
  Eigen::VectorXd beta_new = Rcpp::as<Eigen::VectorXd>(pars_new["beta"]);
  double sigma_new = pars_new["sigma"];

  // compute marginal likelihood ratio: m^((k-k')/2)*(RSS_k/RSS_k')^(m/2)
  double like_ratio = std::pow(m, (k-k_new)/2.0) * std::pow(sigma/sigma_new, m);

  // compute accept probability
  double acc_prob = like_ratio;

  // decide whether to move
  double acc = Rcpp::runif(1, 0.0, 1.0)[0];
  if(acc < acc_prob) {
    k = k_new; xi = xi_new; remain_knots = remain_new;
    beta = beta_new; sigma = sigma_new;
    return true;
  } else {
    return false;
  }
}

void EBARS::_update() {
  bool state = false;
  while(true) {
    state = _jump();
    if(state) {
      break;
    }
  }
}

void EBARS::rjmcmc(int burns, int steps, bool flush, int gap) {
  for(int i=0;i<burns+steps;i++) {
    if(flush) {
      if(i%gap==0) {
        Rcpp::Rcout << "Step " << i << ", RSS = " << (std::pow(sigma,2)*m) << "\n";
        Rcpp::Rcout << k << " knots: " << xi.transpose() << "\n";
      }
    }
    _update();
  }

  if(flush) {
    Rcpp::Rcout << "Step " << (burns+steps) << ", RSS = " << (std::pow(sigma,2)*m) << "\n";
    Rcpp::Rcout << k << " knots: " << xi.transpose() << "\n";
  }
}

Eigen::VectorXd EBARS::predict(const Eigen::VectorXd & x_new) {
  // transform x_new to t_new
  Eigen::VectorXd t_new = (x_new.array()-xmin)/(xmax-xmin);
  // predict
  Eigen::VectorXd y_new = spline_predict(t_new, xi, beta, degree, intercept);
  return y_new;
}

Eigen::VectorXd EBARS::get_knots() {
  return (xi.array()*(xmax-xmin) + xmin);
}


// expose Rcpp class
RCPP_MODULE(class_EBARS) {
  using namespace Rcpp;

  class_<EBARS>("EBARS")

  .constructor<Eigen::VectorXd,Eigen::VectorXd,NumericVector,IntegerVector,List>("constructor")


  .method("rjmcmc", &EBARS::rjmcmc, "reversible jump MCMC")
  .method("predict", &EBARS::predict, "predict by spline regression with EBARS")
  .method("knots", &EBARS::get_knots, "return estimated knots")
  ;
}

RCPP_EXPOSED_CLASS(EBARS)



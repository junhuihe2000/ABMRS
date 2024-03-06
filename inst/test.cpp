bool BinEBARS::_jump_2() {
  double type = Rcpp::runif(1, 0.0, 1.0)[0];
  double birth = _birth_2();
  double death = _death_2();
  if(k_2==1) {death = 0.0;}

  Eigen::VectorXd xi_new, remain_new;
  int k_new;

  // birth scheme
  if(type < birth) {
    k_new = k_2 + 1;
    int idx = Rcpp::sample(n_2-k_2,1)[0] - 1;
    xi_new = Eigen::VectorXd::Zero(k_new);
    remain_new = Eigen::VectorXd::Zero(n_2-k_new);
    xi_new.head(k_2) = xi_2; xi_new(k_2) = remain_knots_2(idx);
    std::sort(xi_new.data(),xi_new.data()+xi_new.size());
    remain_new.head(idx) = remain_knots_2.head(idx);
    remain_new.tail(n_2-k_new-idx) = remain_knots_2.tail(n_2-k_new-idx);
  }

  // death scheme
  else if(type < (birth+death)) {
    k_new = k_2 - 1;
    int idx = Rcpp::sample(k_2,1)[0] - 1;
    xi_new = Eigen::VectorXd::Zero(k_new);
    remain_new = Eigen::VectorXd::Zero(n_2-k_new);
    xi_new.head(idx) = xi_2.head(idx);
    xi_new.tail(k_new-idx) = xi_2.tail(k_new-idx);
    remain_new.head(n_2-k_2) = remain_knots_2; remain_new(n_2-k_2) = xi_2(idx);
  }

  // relocate scheme
  else {
    k_new = k_2;
    int idx_0 = Rcpp::sample(k_2,1)[0] - 1;
    int idx_2 = Rcpp::sample(n_2-k_2,1)[0] - 1;
    xi_new = xi_2; remain_new = remain_knots_2;
    xi_new(idx_0) = remain_knots_2(idx_2);
    remain_new(idx_2) = xi_2(idx_0);
    std::sort(xi_new.data(),xi_new.data()+xi_new.size());
  }

  // compute MLE
  Rcpp::List pars_new = surface_spline_regression(t,y,xi_1,xi_new);
  Eigen::VectorXd beta_new = Rcpp::as<Eigen::VectorXd>(pars_new["beta"]);
  double sigma_new = pars_new["sigma"];

  // compute marginal likelihood ratio: m^(((k+3)(l+3)-(k'+3)(l'+3))/2)*(RSS/RSS')^(m/2)
  double like_ratio = std::pow(m, (k_1+3)*(k_2-k_new)/2.0) * std::pow(sigma/sigma_new, m);

  // compute accept probability
  double acc_prob = like_ratio;

  // decide whether to move
  double acc = Rcpp::runif(1, 0.0, 1.0)[0];
  if(acc < acc_prob) {
    k_2 = k_new; xi_2 = xi_new; remain_knots_2 = remain_new;
    beta = beta_new; sigma = sigma_new;
    return true;
  } else {
    return false;
  }
}

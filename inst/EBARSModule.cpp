#include <Rcpp.h>
using namespace Rcpp ;
#include "ebars.h"

RCPP_MODULE(class_EBARS) {


    class_<EBARS>("EBARS")

    .constructor<const Eigen::VectorXd &,const Eigen::VectorXd,double,double,int,int>()


    .method("rjmcmc", &EBARS::rjmcmc)
    .method("predict", &EBARS::predict)
    ;
}

/*
 * Single File that has all NBGLM and NBGLMM MCMC fitting functions and
 * all other helper functions
 */
#define ARMA_DONT_PRINT_ERRORS
// Header info and Rcpp plugins and depends statements
#include <iostream>
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <math.h>
#include <RcppParallel.h>

using namespace RcppParallel;

//[[Rcpp::plugins("cpp11")]]
//[[Rcpp::depends("RcppArmadillo")]]
//[[Rcpp::depends(RcppParallel)]]

////////////////////////////////////////////////////////////////////////////////
/*
 *Helper functions for doing multivariate normal density evaluation and random draws
 */
////////////////////////////////////////////////////////////////////////////////

const double log2pi = std::log(2.0 * M_PI);

//  Function that computes multivariate normal pdf for a single point
double dmvnrm_1f(const arma::vec &x,
                 const arma::vec &mean,
                 const arma::mat &sigma) {
  int xdim = x.n_elem;
  arma::mat rooti;
  arma::vec z;
  double out;
  double tmp;
  double rootisum, constants;
  constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  arma::mat sigma_copy = sigma;
  // Do the Cholesky decomposition
  arma::mat csigma;
  bool success = false;
  while(success == false)
  {
    success = chol(csigma, sigma_copy, "upper");

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Inv error dmvnorm" << std::endl;
    }
  }
  rooti = arma::trans(arma::inv(trimatu(csigma)));
  rootisum = arma::sum(log(rooti.diag()));
  z = rooti * arma::trans(x.t() - mean.t());
  tmp = constants - 0.5 * arma::sum(z%z) + rootisum;
  out = exp(tmp);
  return(out);
}

double dmvnrm_1f_inv(const arma::vec &x,
                     const arma::vec &mean,
                     const arma::mat &sigma,
                     const arma::mat &sigma_inv){
  double out, denom;
  arma::vec z = x - mean;
  arma::mat num_mat;
  denom = sqrt(arma::det(2.0 * M_PI * sigma));
  num_mat = z.t() * sigma_inv * z;
  out = exp(-num_mat(0, 0) / 2.0) / denom;
  return(out);
}

// Function that generates random multivariate normal points
arma::mat rmvnormal(const int &n,
                    const arma::vec &mu,
                    const arma::mat &sigma) {
  const int d = mu.size();
  arma::vec mu_tmp = mu;

  // Create matrix of standard normal random values
  arma::mat ans(n, d, arma::fill::randn);
  arma::mat sigma_copy = sigma;

  // Do the Cholesky decomposition
  arma::mat csigma;
  bool success = false;

  while(success == false)
  {
    success = chol(csigma, sigma_copy);

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Inv error rmvnorm" << std::endl;
    }
  }

  // Do the transformation
  ans = ans * csigma;
  ans.each_row() += mu.t(); // Add mu to each row in transformed ans

  return ans;
}

// Function that generates a single random multivariate normal point
arma::mat rmvnormal_1(const arma::vec &mu,
                      const arma::mat &sigma) {
  Rcpp::RNGScope();
  const int n = 1;
  const int d = mu.size();
  arma::vec mu_tmp = mu;

  // Create matrix of standard normal random values
  arma::mat ans(n, d, arma::fill::randn);

  arma::mat sigma_copy = sigma;

  // Do the Cholesky decomposition
  arma::mat csigma;
  bool success = false;

  while(success == false)
  {
    success = chol(csigma, sigma_copy);

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Inv error rmvnorm" << std::endl;
    }
  }

  // Do the transformation
  ans = ans * csigma;
  ans.each_row() += mu_tmp.t(); // Add mu to each row in transformed ans

  return ans;
}

double half_cauchy_pdf(const double &x,
                       const double &tau){
  double ret;
  ret = 2.0 / (M_PI * tau) * pow(tau, 2) / (pow(x, 2) + pow(tau, 2));
  return(ret);
}

////////////////////////////////////////////////////////////////////////////////
/*
 * Basic functions for doing NBGLM and NBGLMM MCMC parameter updates
 */
////////////////////////////////////////////////////////////////////////////////

arma::vec update_betas_glm_rw(const arma::rowvec &beta_cur,
                              const arma::rowvec &counts,
                              const double &alpha_cur,
                              const arma::vec &log_offset,
                              const arma::mat &design_mat,
                              const double &prior_sd_betas,
                              const double &rw_var_betas,
                              const int &n_beta,
                              const int &n_sample,
                              int &accept_rec,
                              const int &it_num,
                              const arma::vec &prior_mean_betas){
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec eta(n_sample), eta_prop(n_sample);
  arma::mat R_mat(n_beta, n_beta);
  arma::mat cov_mat_cur(n_beta, n_beta);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  //arma::vec prior_mean_betas(n_beta);
  arma::vec R_mat_diag(n_beta);

  R_mat_diag.fill(prior_var_betas);
  //prior_mean_betas.zeros();
  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  cov_mat_cur.zeros();
  cov_mat_cur.diag() += rw_var_betas;
  beta_prop = arma::trans(rmvnormal(1, beta_cur_tmp, cov_mat_cur));

  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat));

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num < 20){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

arma::vec update_betas_wls_safe(const arma::rowvec &beta_cur,
                                const arma::rowvec &counts,
                                const double &alpha_cur,
                                const arma::vec &log_offset,
                                const arma::mat &design_mat,
                                const double &prior_sd_betas,
                                const int &n_beta,
                                const int &n_sample,
                                const arma::vec &R_mat_diag,
                                int &accept_rec,
                                int &inv_errors,
                                const double &VIF){
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta), mean_wls_prop(n_beta);
  arma::mat R_mat_i(n_beta, n_beta), R_mat(n_beta, n_beta);
  arma::mat cov_mat_cur(n_beta, n_beta), cov_mat_prop(n_beta, n_beta);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  arma::vec prior_mean_betas(n_beta);
  prior_mean_betas.zeros();
  bool success = false;

  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);

  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;
  arma::mat W_mat_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta)));
  success = false;

  R_mat_i.zeros();
  R_mat_i.diag() = 1.0 / R_mat_diag;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  arma::mat ds_W_mat_i = design_mat.t() * W_mat_i;

  arma::mat csigma = R_mat_i + ds_W_mat_i * design_mat;

  success = false;

  success = arma::inv(cov_mat_cur, csigma);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }

  cov_mat_cur = cov_mat_cur * VIF;

  mean_wls_cur = cov_mat_cur * (ds_W_mat_i * y_tilde);

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;

  arma::mat W_mat_prop_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta_prop)));

  arma::mat ds_W_mat_prop_i = design_mat.t() * W_mat_prop_i;
  arma::mat csigma2 = R_mat_i + ds_W_mat_prop_i * design_mat;
  success = false;

  success = arma::inv(cov_mat_prop, csigma2);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }
  cov_mat_prop = cov_mat_prop * VIF;
  mean_wls_prop = cov_mat_prop * (ds_W_mat_prop_i * y_tilde_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f_inv(beta_cur_tmp, prior_mean_betas, R_mat, R_mat_i)) -
    log(dmvnrm_1f_inv(beta_cur_tmp, mean_wls_prop, cov_mat_prop, csigma2));

  mh_prop = ll_prop +
    log(dmvnrm_1f_inv(beta_prop, prior_mean_betas, R_mat, R_mat_i)) -
    log(dmvnrm_1f_inv(beta_prop, mean_wls_cur, cov_mat_cur, csigma));

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

arma::vec update_betas_wls_safe_force(const arma::rowvec &beta_cur,
                                      const arma::rowvec &counts,
                                      const double &alpha_cur,
                                      const arma::vec &log_offset,
                                      const arma::mat &design_mat,
                                      const double &prior_sd_betas,
                                      const int &n_beta,
                                      const int &n_sample,
                                      const arma::vec &R_mat_diag,
                                      int &accept_rec,
                                      int &inv_errors,
                                      const double &VIF,
                                      const arma::vec &prior_mean_betas,
                                      const int &it_num){
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta), mean_wls_prop(n_beta);
  arma::mat R_mat_i(n_beta, n_beta), R_mat(n_beta, n_beta);
  arma::mat cov_mat_cur(n_beta, n_beta), cov_mat_prop(n_beta, n_beta);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  //arma::vec prior_mean_betas(n_beta);
  //prior_mean_betas.zeros();
  bool success = false;

  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);

  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;
  arma::mat W_mat_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta)));
  success = false;

  R_mat_i.zeros();
  R_mat_i.diag() = 1.0 / R_mat_diag;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  arma::mat ds_W_mat_i = design_mat.t() * W_mat_i;

  arma::mat csigma = R_mat_i + ds_W_mat_i * design_mat;

  success = false;

  success = arma::inv(cov_mat_cur, csigma);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }

  //cov_mat_cur = cov_mat_cur * VIF;

  mean_wls_cur = cov_mat_cur * (ds_W_mat_i * y_tilde);

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;

  arma::mat W_mat_prop_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta_prop)));

  arma::mat ds_W_mat_prop_i = design_mat.t() * W_mat_prop_i;
  arma::mat csigma2 = R_mat_i + ds_W_mat_prop_i * design_mat;
  success = false;

  success = arma::inv(cov_mat_prop, csigma2);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }
  //cov_mat_prop = cov_mat_prop * VIF;
  mean_wls_prop = cov_mat_prop * (ds_W_mat_prop_i * y_tilde_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f_inv(beta_cur_tmp, prior_mean_betas, R_mat, R_mat_i)) -
    log(dmvnrm_1f_inv(beta_cur_tmp, mean_wls_prop, cov_mat_prop, csigma2));

  mh_prop = ll_prop +
    log(dmvnrm_1f_inv(beta_prop, prior_mean_betas, R_mat, R_mat_i)) -
    log(dmvnrm_1f_inv(beta_prop, mean_wls_cur, cov_mat_cur, csigma));

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num < 20){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

arma::vec update_betas_wls_mm_force_pb(const arma::rowvec &beta_cur,
                                       const arma::rowvec &counts,
                                       const double &alpha_cur,
                                       const arma::vec &log_offset,
                                       const arma::mat &design_mat,
                                       const double &prior_sd_betas,
                                       const double &re_var,
                                       const int &n_beta,
                                       const int &n_beta_re,
                                       const int &n_sample,
                                       int &accept_rec,
                                       const int &it_num,
                                       const int &num_accept,
                                       int &inv_errors,
                                       const double &prior_mean_b0){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta + n_beta_re), mean_wls_prop(n_beta + n_beta_re);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta + n_beta_re, n_beta + n_beta_re), R_mat_i(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re), cov_mat_prop(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re);
  bool success = false;

  R_mat_diag.zeros();
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;
  prior_mean_betas.zeros();
  prior_mean_betas(0) = prior_mean_b0;
  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;

  arma::mat W_mat_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta)));
  success = false;

  R_mat_i.zeros();
  R_mat_i.diag() = 1.0 / R_mat_diag;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  arma::mat ds_W_mat_i = design_mat.t() * W_mat_i;

  arma::mat csigma = R_mat_i + ds_W_mat_i * design_mat;

  success = false;

  success = arma::inv_sympd(cov_mat_cur, csigma);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }

  mean_wls_cur = cov_mat_cur * (ds_W_mat_i * y_tilde);

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;

  arma::mat W_mat_prop_i = arma::diagmat(1.0 / (alpha_cur + arma::exp(-eta_prop)));

  arma::mat ds_W_mat_prop_i = design_mat.t() * W_mat_prop_i;
  arma::mat csigma2 = R_mat_i + ds_W_mat_prop_i * design_mat;
  success = false;

  success = arma::inv_sympd(cov_mat_prop, csigma2);

  if(success == false)
  {
    inv_errors++;
    return(beta_cur_tmp);
  }

  mean_wls_prop = cov_mat_prop * (ds_W_mat_prop_i * y_tilde_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num <= num_accept){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

arma::vec update_betas_wls_mm_rw(const arma::rowvec &beta_cur,
                                 const arma::rowvec &counts,
                                 const double &alpha_cur,
                                 const arma::vec &log_offset,
                                 const arma::mat &design_mat,
                                 const double &prior_sd_betas,
                                 const double &rw_var_betas,
                                 const double &re_var,
                                 const int &n_beta,
                                 const int &n_beta_re,
                                 const int &n_sample,
                                 int &accept_rec,
                                 const int &it_num,
                                 const arma::vec &prior_mean_betas){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec eta(n_sample), eta_prop(n_sample);
  arma::mat R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec R_mat_diag(n_beta + n_beta_re);
  //arma::vec prior_mean_betas(n_beta + n_beta_re)
  R_mat_diag.zeros();
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;
  //prior_mean_betas.zeros();
  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  cov_mat_cur.zeros();
  cov_mat_cur.diag() += rw_var_betas;
  beta_prop = arma::trans(rmvnormal(1, beta_cur_tmp, cov_mat_cur));

  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat));

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num < 20){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}


//    Function to update NB dispersion parameter for a single feature using
//    a random walk proposal and log-normal prior

double update_rho(const arma::rowvec &beta_cur,
                  const arma::rowvec &counts,
                  const double &alpha_cur,
                  const double &mean_alpha_cur,
                  const arma::vec &log_offset,
                  const arma::mat &design_mat,
                  const double &prior_sd_rs,
                  const double &rw_sd_rs,
                  const int &n_beta,
                  const int &n_sample,
                  int &n_accept){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;

  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset);

  alpha_prop = exp(log(alpha_cur) + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;

  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));

  mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  //if((R::runif(0, 1) < exp(mh_prop - mh_cur)) && 1 < 0){
  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    n_accept++;
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}

double update_rho_hc(const arma::rowvec &beta_cur,
                     const arma::rowvec &counts,
                     const double &alpha_cur,
                     const double &mean_alpha_cur,
                     const arma::vec &log_offset,
                     const arma::mat &design_mat,
                     const double &prior_sd_rs,
                     const double &rw_sd_rs,
                     const int &n_beta,
                     const int &n_sample,
                     int &n_accept){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;
  double tau = 20;
  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset);

  alpha_prop = fabs(alpha_cur + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;

  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));

  // mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  // mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  mh_cur = ll_cur + half_cauchy_pdf(alpha_cur, tau);
  mh_prop = ll_prop + half_cauchy_pdf(alpha_prop, tau);

  //if((R::runif(0, 1) < exp(mh_prop - mh_cur)) && 1 < 0){
  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    n_accept++;
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}

//    Function to update NB dispersion parameter for a single feature using
//    a random walk proposal and log-normal prior (forced non-accepts for num_accept it)

double update_rho_force(const arma::rowvec &beta_cur,
                        const arma::rowvec &counts,
                        const double &alpha_cur,
                        const double &mean_alpha_cur,
                        const arma::vec &log_offset,
                        const arma::mat &design_mat,
                        const double &prior_sd_rs,
                        const double &rw_sd_rs,
                        const int &n_beta,
                        const int &n_sample,
                        const int &it_num,
                        const int &num_accept,
                        int &n_accept){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;

  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset);

  alpha_prop = exp(log(alpha_cur) + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;

  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));

  mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) && it_num > num_accept){
    n_accept++;
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}

//    Function to update NB dispersion parameter for a single feature using
//    a random walk proposal and log-normal prior (forced non-accepts for num_accept it)

double update_rho_force_hc(const arma::rowvec &beta_cur,
                           const arma::rowvec &counts,
                           const double &alpha_cur,
                           const double &mean_alpha_cur,
                           const arma::vec &log_offset,
                           const arma::mat &design_mat,
                           const double &prior_sd_rs,
                           const double &rw_sd_rs,
                           const int &n_beta,
                           const int &n_sample,
                           const int &it_num,
                           const int &num_accept,
                           int &n_accept){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;
  double tau = 20;
  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset);

  alpha_prop = fabs(alpha_cur + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;

  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));

  // mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  // mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  mh_cur = ll_cur + log(half_cauchy_pdf(alpha_cur, tau));
  mh_prop = ll_prop + log(half_cauchy_pdf(alpha_prop, tau));

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) && it_num > num_accept){
    n_accept++;
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////     NBGLM Random Walk Version                                       //////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



arma::mat whole_chain_nbglm_rw(const arma::rowvec &counts,
                               const arma::vec &log_offset,
                               const arma::rowvec &starting_betas,
                               const arma::mat &design_mat,
                               const arma::mat &contrast_mat,
                               const double &mean_rho,
                               const double &prior_sd_rs,
                               const double &rw_sd_rs,
                               const double &rw_var_betas,
                               const double &prior_sd_betas,
                               const double &n_beta,
                               const double &n_sample,
                               const int n_it,
                               const double &prop_burn,
                               const bool &return_cont){
  int n_beta_tot = n_beta, accepts = 0, accepts_alpha = 0, i, num_accept = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat ret(n_beta + n_cont, 9, arma::fill::zeros), contrast_sample(n_it, n_cont);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it);
  arma::vec prior_mean_betas(n_beta);
  prior_mean_betas.zeros();
  prior_mean_betas(0) = log(mean(counts));

  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  betas_cur = starting_betas;
  betas_last = starting_betas;

  for(i = 1; i < n_it; i++){
    betas_cur = arma::trans(update_betas_glm_rw(betas_last,
                                                counts,
                                                disp_sample(i-1),
                                                log_offset,
                                                design_mat,
                                                prior_sd_betas,
                                                rw_var_betas,
                                                n_beta,
                                                n_sample,
                                                accepts,
                                                i,
                                                prior_mean_betas));
    betas_last = betas_cur;
    betas_sample.row(i) = betas_cur;

    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 4) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 5) = accepts;
  ret(0, 6) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    //ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 2) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 3) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
  }
  // Calculating Geweke p-values for regression coefficients
  arma::mat betas_nb = betas_sample.rows(burn_bound, n_it - 1);
  arma::vec disp_nb = disp_sample.rows(burn_bound, n_it - 1);
  int n_row_nb = betas_nb.n_rows, n_thin = 40;
  arma::uvec idx_thin = arma::regspace<arma::uvec>(0, n_thin, n_row_nb-1);

  betas_nb = betas_nb.rows(idx_thin);
  disp_nb = arma::log(disp_nb.rows(idx_thin));
  n_row_nb = betas_nb.n_rows;

  int gub = round(n_row_nb * 0.2);
  int glb = round(n_row_nb * 0.5);
  double var_first, var_second, mean_first, mean_second, z_g;
  double df_t;

  for(int kk = 0; kk < n_beta; kk++){
    var_first = arma::var(betas_nb.rows(0, gub - 1).col(kk)) / gub;
    var_second = arma::var(betas_nb.rows(glb - 1, n_row_nb - 1).col(kk)) / (n_row_nb / 2.0);
    mean_first = arma::mean(betas_nb.rows(0, gub - 1).col(kk));
    mean_second = arma::mean(betas_nb.rows(glb, n_row_nb - 1).col(kk));
    z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
    df_t = pow(var_first + var_second, 2) /
      (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
    //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
    ret(kk, 7) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  }

  // Calculating Geweke p-values for dispersion
  var_first = arma::var(disp_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(disp_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(disp_nb.rows(0, gub - 1));
  mean_second = arma::mean(disp_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 8) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  return(ret);
}


struct whole_feature_sample_rw_struct_glm : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const double &rw_var_betas;
  const int &n_beta;
  const int &n_sample;
  const int &n_it;
  const double &prop_burn;
  const bool &return_cont;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_rw_struct_glm(const arma::mat &counts,
                                     const arma::vec &log_offset,
                                     const arma::mat &starting_betas,
                                     const arma::mat &design_mat,
                                     const arma::mat &contrast_mat,
                                     const arma::vec &mean_rhos,
                                     const double &prior_sd_rs,
                                     const double &rw_sd_rs,
                                     const double &prior_sd_betas,
                                     const double &rw_var_betas,
                                     const int &n_beta,
                                     const int &n_sample,
                                     const int &n_it,
                                     const double &prop_burn,
                                     const bool &return_cont,
                                     arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), rw_var_betas(rw_var_betas), n_beta(n_beta),
      n_sample(n_sample), n_it(n_it), prop_burn(prop_burn), return_cont(return_cont), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglm_rw(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      rw_var_betas,
                      prior_sd_betas,
                      n_beta,
                      n_sample,
                      n_it,
                      prop_burn,
                      return_cont);
    }
  }

};

arma::cube mcmc_chain_glm_rw_par(const arma::mat &counts,
                                 const arma::vec &log_offset,
                                 const arma::mat &starting_betas,
                                 const arma::mat &design_mat,
                                 const arma::mat &contrast_mat,
                                 const arma::vec &mean_rhos,
                                 const double &prior_sd_rs,
                                 const double &rw_sd_rs,
                                 const double &rw_var_betas,
                                 const double &prior_sd_betas,
                                 const int &n_beta,
                                 const int &n_sample,
                                 const int &n_it,
                                 const double &prop_burn,
                                 const bool &return_cont){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 9, counts.n_rows, arma::fill::zeros);

  whole_feature_sample_rw_struct_glm mcmc_inst(counts,
                                               log_offset,
                                               starting_betas,
                                               design_mat,
                                               contrast_mat,
                                               mean_rhos,
                                               prior_sd_rs,
                                               rw_sd_rs,
                                               prior_sd_betas,
                                               rw_var_betas,
                                               n_beta,
                                               n_sample,
                                               n_it,
                                               prop_burn,
                                               return_cont,
                                               upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}


///' Negative Binomial GLMM MCMC Random Walk (full parallel chians)
///'
///' Run an MCMC for the Negative Binomial mixed model (short description, one or two sentences)
///'
///' This is where you write details on the function...
///'
///' more details....
///'
///' @param counts a matrix of counts
///' @param design_mat design matrix for mean response
///' @param design_mat_re design matrix for random intercepts
///' @param prior_sd_betas prior std. dev. for regression coefficients
///' @param rw_sd_betas random walk std. dev. for proposing beta values
///' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
///' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
///' @param prior_sd_rs prior std. dev for dispersion parameters
///' @param prior_mean_log_rs vector of prior means for dispersion parameters
///' @param n_it number of iterations to run MCMC
///' @param rw_sd_rs random walk std. dev. for proposing dispersion values
///' @param log_offset vector of offsets on log scale
///'
///' @author Brian Vestal
///'
///' @return
///' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
///'
///' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc_rw(arma::mat counts,
                         arma::mat design_mat,
                         arma::mat contrast_mat,
                         double prior_sd_betas,
                         double rw_sd_betas,
                         double prior_sd_rs,
                         arma::vec prior_mean_log_rs,
                         int n_it,
                         double rw_sd_rs,
                         arma::vec log_offset,
                         arma::mat starting_betas,
                         double prop_burn = .10,
                         bool return_cont = false,
                         Rcpp::StringVector beta_names = NA_STRING,
                         Rcpp::StringVector cont_names = NA_STRING){

  //int grain_size = 1;
  arma::cube ret;
  arma::mat design_mat_tot = design_mat;
  int n_beta = design_mat.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  double rw_var_betas = pow(rw_sd_betas, 2);
  arma::mat starting_betas2(counts.n_rows, n_beta), contrast_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_glm_rw_par(counts,
                              log_offset,
                              starting_betas2,
                              design_mat_tot,
                              contrast_mat_trans,
                              prior_mean_log_rs,
                              prior_sd_rs,
                              rw_sd_rs,
                              rw_var_betas,
                              prior_sd_betas,
                              n_beta,
                              n_sample,
                              n_it,
                              prop_burn,
                              return_cont);
  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alphas;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 3));
  disp_ret = ret.tube(0, 4);
  accepts_ret = ret.tube(0, 5);
  accepts_ret_alphas = ret.tube(0, 6);
  //inv_errors_ret = ret.tube(0, n_beta+2);

  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev",
                                                              "BF_exact", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;

  arma::mat geweke_ret_beta, geweke_ret_alpha;

  geweke_ret_beta = ret.tube(arma::span(0, n_beta - 1), arma::span(7));
  geweke_ret_alpha = ret.tube(0, 8);
  arma::mat ret_gwe = arma::join_horiz(geweke_ret_beta.t(), geweke_ret_alpha.t());

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 3));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);
    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alphas,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alphas,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }

}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/////////////////////////    NBGLMM RW Version      ///////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_rw2(const arma::rowvec &counts,
                                 const arma::vec &log_offset,
                                 const arma::rowvec &starting_betas,
                                 const arma::mat &design_mat,
                                 const arma::mat &contrast_mat,
                                 const double &mean_rho,
                                 const double &prior_sd_rs,
                                 const double &rw_sd_rs,
                                 const double &rw_var_betas,
                                 const double &prior_sd_betas,
                                 const double &n_beta,
                                 const double &n_beta_re,
                                 const double &n_sample,
                                 const double &prior_sd_betas_a,
                                 const double &prior_sd_betas_b,
                                 const int n_it,
                                 const double &prop_burn,
                                 const bool &return_cont){
  int n_beta_tot = n_beta + n_beta_re, accepts = 0, accepts_alpha = 0, i, num_accept = 0, n_cont = 0;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat ret(n_beta + n_cont, 11, arma::fill::zeros), contrast_sample(n_it, n_cont);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  arma::vec prior_mean_betas(n_beta + n_beta_re);
  prior_mean_betas.zeros();
  prior_mean_betas(0) = log(mean(counts));

  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;

  for(i = 1; i < n_it; i++){
    betas_cur = arma::trans(update_betas_wls_mm_rw(betas_last,
                                                   counts,
                                                   disp_sample(i-1),
                                                   log_offset,
                                                   design_mat,
                                                   prior_sd_betas,
                                                   rw_var_betas,
                                                   sigma2_sample(i-1),
                                                   n_beta,
                                                   n_beta_re,
                                                   n_sample,
                                                   accepts,
                                                   i,
                                                   prior_mean_betas));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);

    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);
    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 4) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 5) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 6) = accepts;
  ret(0, 7) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    //ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 2) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 4) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
  }

  // Calculating Geweke p-values for regression coefficients
  arma::mat betas_nb = betas_sample.rows(burn_bound, n_it - 1);
  arma::vec disp_nb = disp_sample.rows(burn_bound, n_it - 1);
  arma::vec sigma2_nb = sigma2_sample.rows(burn_bound, n_it - 1);
  int n_row_nb = betas_nb.n_rows, n_thin = 40;
  arma::uvec idx_thin = arma::regspace<arma::uvec>(0, n_thin, n_row_nb-1);

  betas_nb = betas_nb.rows(idx_thin);
  disp_nb = arma::log(disp_nb.rows(idx_thin));
  sigma2_nb = arma::log(sigma2_nb.rows(idx_thin));
  n_row_nb = betas_nb.n_rows;

  int gub = round(n_row_nb * 0.2);
  int glb = round(n_row_nb * 0.5);

  double df_t;
  double var_first, var_second, mean_first, mean_second, z_g;

  for(int kk = 0; kk < n_beta; kk++){
    var_first = arma::var(betas_nb.rows(0, gub - 1).col(kk)) / gub;
    var_second = arma::var(betas_nb.rows(glb - 1, n_row_nb - 1).col(kk)) / (n_row_nb / 2.0);
    mean_first = arma::mean(betas_nb.rows(0, gub - 1).col(kk));
    mean_second = arma::mean(betas_nb.rows(glb, n_row_nb - 1).col(kk));
    z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
    df_t = pow(var_first + var_second, 2) /
      (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
    //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
    ret(kk, 8) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  }

  // Calculating Geweke p-values for dispersion
  var_first = arma::var(disp_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(disp_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(disp_nb.rows(0, gub - 1));
  mean_second = arma::mean(disp_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 9) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  // Calculating Geweke p-values for RI variance
  var_first = arma::var(sigma2_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(sigma2_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(sigma2_nb.rows(0, gub - 1));
  mean_second = arma::mean(sigma2_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 10) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  return(ret);
}



struct whole_feature_sample_rw_struct2 : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const double &rw_var_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const double &prop_burn;
  const bool &return_cont;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_rw_struct2(const arma::mat &counts,
                                  const arma::vec &log_offset,
                                  const arma::mat &starting_betas,
                                  const arma::mat &design_mat,
                                  const arma::mat &contrast_mat,
                                  const arma::vec &mean_rhos,
                                  const double &prior_sd_rs,
                                  const double &rw_sd_rs,
                                  const double &prior_sd_betas,
                                  const double &rw_var_betas,
                                  const int &n_beta,
                                  const int &n_beta_re,
                                  const int &n_sample,
                                  const double &prior_sd_betas_a,
                                  const double &prior_sd_betas_b,
                                  const int &n_it,
                                  const double &prop_burn,
                                  const bool &return_cont,
                                  arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), rw_var_betas(rw_var_betas), n_beta(n_beta), n_beta_re(n_beta_re),
      n_sample(n_sample), prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it),
      prop_burn(prop_burn), return_cont(return_cont), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_rw2(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      rw_var_betas,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      prop_burn,
                      return_cont);
    }
  }

};

arma::cube mcmc_chain_rw_par2(const arma::mat &counts,
                              const arma::vec &log_offset,
                              const arma::mat &starting_betas,
                              const arma::mat &design_mat,
                              const arma::mat &contrast_mat,
                              const arma::vec &mean_rhos,
                              const double &prior_sd_rs,
                              const double &rw_sd_rs,
                              const double &rw_var_betas,
                              const double &prior_sd_betas,
                              const int &n_beta,
                              const int &n_beta_re,
                              const int &n_sample,
                              const double &prior_sd_betas_a,
                              const double &prior_sd_betas_b,
                              const int &n_it,
                              const double &prop_burn,
                              const bool &return_cont){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 11, counts.n_rows, arma::fill::zeros);

  whole_feature_sample_rw_struct2 mcmc_inst(counts,
                                            log_offset,
                                            starting_betas,
                                            design_mat,
                                            contrast_mat,
                                            mean_rhos,
                                            prior_sd_rs,
                                            rw_sd_rs,
                                            prior_sd_betas,
                                            rw_var_betas,
                                            n_beta,
                                            n_beta_re,
                                            n_sample,
                                            prior_sd_betas_a,
                                            prior_sd_betas_b,
                                            n_it,
                                            prop_burn,
                                            return_cont,
                                            upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}


// ' Negative Binomial GLMM MCMC Random Walk (full parallel chians)
// '
// ' Run an MCMC for the Negative Binomial mixed model (short description, one or two sentences)
// '
// ' This is where you write details on the function...
// '
// ' more details....
// '
// ' @param counts a matrix of counts
// ' @param design_mat design matrix for mean response
// ' @param design_mat_re design matrix for random intercepts
// ' @param prior_sd_betas prior std. dev. for regression coefficients
// ' @param rw_sd_betas random walk std. dev. for proposing beta values
// ' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
// ' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
// ' @param prior_sd_rs prior std. dev for dispersion parameters
// ' @param prior_mean_log_rs vector of prior means for dispersion parameters
// ' @param n_it number of iterations to run MCMC
// ' @param rw_sd_rs random walk std. dev. for proposing dispersion values
// ' @param log_offset vector of offsets on log scale
// '
// ' @author Brian Vestal
// '
// ' @return
// ' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
// '
// ' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_rw(arma::mat counts,
                          arma::mat design_mat,
                          arma::mat design_mat_re,
                          arma::mat contrast_mat,
                          double prior_sd_betas,
                          double rw_sd_betas,
                          double prior_sd_betas_a,
                          double prior_sd_betas_b,
                          double prior_sd_rs,
                          arma::vec prior_mean_log_rs,
                          int n_it,
                          double rw_sd_rs,
                          arma::vec log_offset,
                          arma::mat starting_betas,
                          double prop_burn = .10,
                          bool return_cont = false,
                          Rcpp::StringVector beta_names = NA_STRING,
                          Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  double rw_var_betas = pow(rw_sd_betas, 2);
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), contrast_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_rw_par2(counts,
                           log_offset,
                           starting_betas2,
                           design_mat_tot,
                           contrast_mat_trans,
                           prior_mean_log_rs,
                           prior_sd_rs,
                           rw_sd_rs,
                           rw_var_betas,
                           prior_sd_betas,
                           n_beta,
                           n_beta_re,
                           n_sample,
                           prior_sd_betas_a,
                           prior_sd_betas_b,
                           n_it,
                           prop_burn,
                           return_cont);
  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alpha;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 3));
  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev",
                                                              "BF_exact", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;
  disp_ret = ret.tube(0, 4);
  sigma2_ret = ret.tube(0, 5);
  accepts_ret = ret.tube(0, 6);
  accepts_ret_alpha = ret.tube(0, 7);

  arma::mat geweke_ret_beta, geweke_ret_alpha, geweke_ret_sig2;
  geweke_ret_beta = ret.tube(arma::span(0, n_beta - 1), arma::span(8));
  geweke_ret_alpha = ret.tube(0, 9);
  geweke_ret_sig2 = ret.tube(0, 10);
  arma::mat ret_gwe = arma::join_horiz(geweke_ret_beta.t(), geweke_ret_alpha.t());
  ret_gwe = arma::join_horiz(ret_gwe, geweke_ret_sig2.t());

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 3));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);

    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//////////////     NBGLM WLS         ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


arma::mat whole_chain_nbglm_sum_cont(const arma::rowvec &counts,
                                     const arma::vec &log_offset,
                                     const arma::rowvec &starting_betas,
                                     const arma::mat &design_mat,
                                     const arma::mat &contrast_mat,
                                     const double &mean_rho,
                                     const double &prior_sd_betas,
                                     const double &prior_sd_rs,
                                     const double &rw_sd_rs,
                                     const double &n_beta,
                                     const double &n_sample,
                                     const int n_it,
                                     const double &prop_burn,
                                     const bool &return_cont){
  int i = 1, accepts = 0, accepts_alphas = 0, inv_errors = 0, n_cont = 0, num_accept = 20;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  double VIF = 1;
  //arma::mat ret(n_it, n_beta + 3, arma::fill::zeros);
  arma::mat ret(n_beta + n_cont, 9, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta), contrast_sample(n_it, n_cont);
  arma::rowvec betas_cur(n_beta), betas_last(n_beta);
  arma::vec disp_sample(n_it);
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec R_mat_diag(n_beta);
  R_mat_diag.fill(prior_var_betas);
  arma::vec prior_mean_betas(n_beta);
  prior_mean_betas.zeros();
  prior_mean_betas(0) = log(mean(counts));

  betas_sample.row(0) = starting_betas;
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  betas_cur = starting_betas;
  betas_last = starting_betas;

  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_safe_force(betas_last,
                                                        counts,
                                                        disp_sample(i-1),
                                                        log_offset,
                                                        design_mat,
                                                        prior_sd_betas,
                                                        n_beta,
                                                        n_sample,
                                                        R_mat_diag,
                                                        accepts,
                                                        inv_errors,
                                                        VIF,
                                                        prior_mean_betas,
                                                        i));
    betas_last = betas_cur;
    betas_sample.row(i) = betas_cur;
    /*
     * (betas_cur,
     counts,
    disp_sample(i-1),
    mean_rho,
    log_offset,
    design_mat,
    prior_sd_rs,
    rw_sd_rs,
    n_beta_tot,
    n_sample,
    i,
    num_accept,
    accepts_alpha);
     */

    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta,
                n_sample,
                i,
                num_accept,
                accepts_alphas);
    i++;
  }
  if(inv_errors > 0){
    betas_sample.fill(NA_REAL);
    disp_sample.fill(NA_REAL);
    accepts = -1;
  }

  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 4) = arma::mean(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 5) = accepts;
  ret(0, 6) = accepts_alphas;
  for(int k = 0; k < n_beta + n_cont; k++){
    //ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 2) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 3) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);

  }

  // Calculating Geweke p-values for regression coefficients
  arma::mat betas_nb = betas_sample.rows(burn_bound, n_it - 1);
  arma::vec disp_nb = disp_sample.rows(burn_bound, n_it - 1);
  int n_row_nb = betas_nb.n_rows, n_thin = 40;
  arma::uvec idx_thin = arma::regspace<arma::uvec>(0, n_thin, n_row_nb-1);

  betas_nb = betas_nb.rows(idx_thin);
  disp_nb = arma::log(disp_nb.rows(idx_thin));
  n_row_nb = betas_nb.n_rows;

  int gub = round(n_row_nb * 0.2);
  int glb = round(n_row_nb * 0.5);
  double var_first, var_second, mean_first, mean_second, z_g;
  double df_t;

  for(int kk = 0; kk < n_beta; kk++){
    var_first = arma::var(betas_nb.rows(0, gub - 1).col(kk)) / gub;
    var_second = arma::var(betas_nb.rows(glb - 1, n_row_nb - 1).col(kk)) / (n_row_nb / 2.0);
    mean_first = arma::mean(betas_nb.rows(0, gub - 1).col(kk));
    mean_second = arma::mean(betas_nb.rows(glb, n_row_nb - 1).col(kk));
    z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
    df_t = pow(var_first + var_second, 2) /
      (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
    //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
    ret(kk, 7) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  }

  // Calculating Geweke p-values for dispersion
  var_first = arma::var(disp_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(disp_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(disp_nb.rows(0, gub - 1));
  mean_second = arma::mean(disp_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 8) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  return(ret);
}

struct whole_feature_sample_struct_glm_sum_cont : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_sample;
  const int &n_it;
  const double &prop_burn;
  const bool &return_cont;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_glm_sum_cont(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::mat &contrast_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_sample,
                                           const int &n_it,
                                           const double &prop_burn,
                                           const bool &return_cont,
                                           arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_sample(n_sample), n_it(n_it),
      prop_burn(prop_burn), return_cont(return_cont), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglm_sum_cont(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_betas,
                      prior_sd_rs,
                      rw_sd_rs,
                      n_beta,
                      n_sample,
                      n_it,
                      prop_burn,
                      return_cont);
    }
  }

};

arma::cube mcmc_chain_glm_sum_cont_par(const arma::mat &counts,
                                       const arma::vec &log_offset,
                                       const arma::mat &starting_betas,
                                       const arma::mat &design_mat,
                                       const arma::mat &contrast_mat,
                                       const arma::vec &mean_rhos,
                                       const double &prior_sd_rs,
                                       const double &rw_sd_rs,
                                       const double &prior_sd_betas,
                                       const int &n_beta,
                                       const int &n_sample,
                                       const int &n_it,
                                       const double &prop_burn,
                                       const bool &return_cont){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 9, counts.n_rows, arma::fill::zeros);

  whole_feature_sample_struct_glm_sum_cont mcmc_inst(counts,
                                                     log_offset,
                                                     starting_betas,
                                                     design_mat,
                                                     contrast_mat,
                                                     mean_rhos,
                                                     prior_sd_rs,
                                                     rw_sd_rs,
                                                     prior_sd_betas,
                                                     n_beta,
                                                     n_sample,
                                                     n_it,
                                                     prop_burn,
                                                     return_cont,
                                                     upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  return(upd_param);
}

// ' Negative Binomial GLM MCMC WLS (full parallel chians)
// '
// ' Run an MCMC for the Negative Binomial mixed model (short description, one or two sentences)
// '
// ' This is where you write details on the function...
// '
// ' more details....
// '
// ' @param counts a matrix of counts
// ' @param design_mat design matrix for mean response
// ' @param contrast_mat contrast matrix (each row is a contrast of regression parameters to be tested)
// ' @param prior_sd_betas prior std. dev. for regression coefficients
// ' @param prior_sd_rs prior std. dev for dispersion parameters
// ' @param prior_mean_log_rs vector of prior means for dispersion parameters
// ' @param n_it number of iterations to run MCMC
// ' @param rw_sd_rs random wal std. dev. for proposing dispersion values
// ' @param log_offset vector of offsets on log scale
// '
// ' @author Brian Vestal
// '
// ' @return
// ' Returns a list with a cube of regression parameters, and a matrix of dispersion values
// '
// //' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc_wls(arma::mat counts,
                          arma::mat design_mat,
                          arma::mat contrast_mat,
                          double prior_sd_betas,
                          double prior_sd_rs,
                          arma::vec prior_mean_log_rs,
                          int n_it,
                          double rw_sd_rs,
                          arma::vec log_offset,
                          arma::mat starting_betas,
                          double burn_in_prop = .1,
                          bool return_cont = false,
                          Rcpp::StringVector beta_names = NA_STRING,
                          Rcpp::StringVector cont_names = NA_STRING){

  int n_beta = design_mat.n_cols, n_sample = counts.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::cube ret;
  int n_beta_start = starting_betas.n_cols;
  arma::mat starting_betas2(counts.n_rows, n_beta), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;

  ret = mcmc_chain_glm_sum_cont_par(counts,
                                    log_offset,
                                    starting_betas2,
                                    design_mat,
                                    cont_mat_trans,
                                    prior_mean_log_rs,
                                    prior_sd_rs,
                                    rw_sd_rs,
                                    prior_sd_betas,
                                    n_beta,
                                    n_sample,
                                    n_it,
                                    burn_in_prop,
                                    return_cont);


  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alphas;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 3));
  disp_ret = ret.tube(0, 4);
  accepts_ret = ret.tube(0, 5);
  accepts_ret_alphas = ret.tube(0, 6);
  //inv_errors_ret = ret.tube(0, n_beta+2);

  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev",
                                                              "BF_exact", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;

  arma::mat geweke_ret_beta, geweke_ret_alpha;

  geweke_ret_beta = ret.tube(arma::span(0, n_beta - 1), arma::span(7));
  geweke_ret_alpha = ret.tube(0, 8);
  arma::mat ret_gwe = arma::join_horiz(geweke_ret_beta.t(), geweke_ret_alpha.t());

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 3));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);

    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alphas,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alphas,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }

}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/////     NBGLMM Summary version (with Contrasts)     /////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


double update_sigma2_rw(const double &sigma2_cur,
                        const double &sd_rw_sigma,
                        const arma::rowvec &rand_int_vec){
  double ll_cur, ll_prop, sigma2_prop, mh_prop, mh_cur, sigma_cur, sigma_prop;
  arma::rowvec pdf_vals_cur, pdf_vals_prop;

  sigma_cur = sqrt(sigma2_cur);
  sigma2_prop = fabs(sigma2_cur + R::rnorm(0, sd_rw_sigma));
  sigma_prop = sqrt(sigma2_prop);
  pdf_vals_cur = arma::normpdf(rand_int_vec, 0, sigma_cur);
  pdf_vals_prop = arma::normpdf(rand_int_vec, 0, sigma_prop);
  ll_cur = arma::sum(arma::log(pdf_vals_cur));
  ll_prop = arma::sum(arma::log(pdf_vals_prop));
  mh_cur = ll_cur - log(sigma_cur);
  mh_prop = ll_prop - log(sigma_prop);
  // mh_cur = ll_cur;
  // mh_prop = ll_prop;
  if((R::runif(0, 1) < exp(mh_prop - mh_cur))){
    return sigma2_prop;
  }
  else{
    return sigma2_cur;
  }
}

double update_sigma2_rw_hc(const double &sigma2_cur,
                           const double &sd_rw_sigma,
                           const arma::rowvec &rand_int_vec,
                           const double &tau){
  double ll_cur, ll_prop, sigma2_prop, mh_prop, mh_cur, sigma_cur, sigma_prop;
  arma::rowvec pdf_vals_cur, pdf_vals_prop;

  sigma_cur = sqrt(sigma2_cur);
  sigma2_prop = fabs(sigma2_cur + R::rnorm(0, sd_rw_sigma));
  sigma_prop = sqrt(sigma2_prop);
  pdf_vals_cur = arma::normpdf(rand_int_vec, 0, sigma_cur);
  pdf_vals_prop = arma::normpdf(rand_int_vec, 0, sigma_prop);
  ll_cur = arma::sum(arma::log(pdf_vals_cur));
  ll_prop = arma::sum(arma::log(pdf_vals_prop));
  mh_cur = ll_cur + log(half_cauchy_pdf(sigma_cur, tau));
  mh_prop = ll_prop + log(half_cauchy_pdf(sigma_prop, tau));
  // mh_cur = ll_cur;
  // mh_prop = ll_prop;
  if((R::runif(0, 1) < exp(mh_prop - mh_cur))){
    return sigma2_prop;
  }
  else{
    return sigma2_cur;
  }
}


//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_sum_cont_pb(const arma::rowvec &counts,
                                         const arma::vec &log_offset,
                                         const arma::rowvec &starting_betas,
                                         const arma::mat &design_mat,
                                         const arma::mat &contrast_mat,
                                         const double &mean_rho,
                                         const double &prior_sd_rs,
                                         const double &rw_sd_rs,
                                         const double &prior_sd_betas,
                                         const double &n_beta,
                                         const double &n_beta_re,
                                         const double &n_sample,
                                         const double &prior_sd_betas_a,
                                         const double &prior_sd_betas_b,
                                         const int &n_it,
                                         const int &num_accept,
                                         const double &prop_burn,
                                         const bool &return_cont){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, accepts_alpha = 0, inv_errors = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat contrast_sample;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_beta + n_cont, 11, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  double beta0_prior_mean = log(arma::mean(counts));
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;
  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force_pb(betas_last,
                                                         counts,
                                                         disp_sample(i-1),
                                                         log_offset,
                                                         design_mat,
                                                         prior_sd_betas,
                                                         sigma2_sample(i-1),
                                                         n_beta,
                                                         n_beta_re,
                                                         n_sample,
                                                         accepts,
                                                         i,
                                                         num_accept,
                                                         inv_errors,
                                                         beta0_prior_mean));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);

    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    i++;
  }
  if(inv_errors > 0){
    ret.fill(NA_REAL);
    accepts = -1;
    ret(0, 7) = accepts;
    return(ret);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  double var_first, var_second, mean_first, mean_second, z_g;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 4) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 5) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 6) = accepts;
  ret(0, 7) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    //ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 2) =  arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 3) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
  }

  // Calculating Geweke p-values for regression coefficients
  arma::mat betas_nb = betas_sample.rows(burn_bound, n_it - 1);
  arma::vec disp_nb = disp_sample.rows(burn_bound, n_it - 1);
  arma::vec sigma2_nb = sigma2_sample.rows(burn_bound, n_it - 1);
  int n_row_nb = betas_nb.n_rows, n_thin = 40;
  arma::uvec idx_thin = arma::regspace<arma::uvec>(0, n_thin, n_row_nb-1);

  betas_nb = betas_nb.rows(idx_thin);
  disp_nb = arma::log(disp_nb.rows(idx_thin));
  sigma2_nb = arma::log(sigma2_nb.rows(idx_thin));
  n_row_nb = betas_nb.n_rows;

  int gub = round(n_row_nb * 0.2);
  int glb = round(n_row_nb * 0.5);

  double df_t;

  for(int kk = 0; kk < n_beta; kk++){
    var_first = arma::var(betas_nb.rows(0, gub - 1).col(kk)) / gub;
    var_second = arma::var(betas_nb.rows(glb - 1, n_row_nb - 1).col(kk)) / (n_row_nb / 2.0);
    mean_first = arma::mean(betas_nb.rows(0, gub - 1).col(kk));
    mean_second = arma::mean(betas_nb.rows(glb, n_row_nb - 1).col(kk));
    z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
    df_t = pow(var_first + var_second, 2) /
      (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
    //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
    ret(kk, 8) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  }

  // Calculating Geweke p-values for dispersion
  var_first = arma::var(disp_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(disp_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(disp_nb.rows(0, gub - 1));
  mean_second = arma::mean(disp_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 9) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  // Calculating Geweke p-values for RI variance
  var_first = arma::var(sigma2_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(sigma2_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(sigma2_nb.rows(0, gub - 1));
  mean_second = arma::mean(sigma2_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 10) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  return(ret);
}

struct whole_feature_sample_struct_sum_cont_pb : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const int &num_accept;
  const double &prop_burn;
  const bool &return_cont;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_sum_cont_pb(const arma::mat &counts,
                                          const arma::vec &log_offset,
                                          const arma::mat &starting_betas,
                                          const arma::mat &design_mat,
                                          const arma::mat &contrast_mat,
                                          const arma::vec &mean_rhos,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const int &n_beta,
                                          const int &n_beta_re,
                                          const int &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it,
                                          const int &num_accept,
                                          const double &prop_burn,
                                          const bool &return_cont,
                                          arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), return_cont(return_cont), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_sum_cont_pb(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      num_accept,
                      prop_burn,
                      return_cont);
    }
  }

};

arma::cube mcmc_chain_par_sum_cont_pb(const arma::mat &counts,
                                      const arma::vec &log_offset,
                                      const arma::mat &starting_betas,
                                      const arma::mat &design_mat,
                                      const arma::mat &contrast_mat,
                                      const arma::vec &mean_rhos,
                                      const double &prior_sd_rs,
                                      const double &rw_sd_rs,
                                      const double &prior_sd_betas,
                                      const int &n_beta,
                                      const int &n_beta_re,
                                      const int &n_sample,
                                      const double &prior_sd_betas_a,
                                      const double &prior_sd_betas_b,
                                      const int &n_it,
                                      const int &num_accept,
                                      const double &prop_burn,
                                      const bool &return_cont){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 11, counts.n_rows, arma::fill::zeros);
  whole_feature_sample_struct_sum_cont_pb mcmc_inst(counts,
                                                    log_offset,
                                                    starting_betas,
                                                    design_mat,
                                                    contrast_mat,
                                                    mean_rhos,
                                                    prior_sd_rs,
                                                    rw_sd_rs,
                                                    prior_sd_betas,
                                                    n_beta,
                                                    n_beta_re,
                                                    n_sample,
                                                    prior_sd_betas_a,
                                                    prior_sd_betas_b,
                                                    n_it,
                                                    num_accept,
                                                    prop_burn,
                                                    return_cont,
                                                    upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

// ' Negative Binomial GLMM fit for RNA-Seq expression using MCMC (contrast version)
// '
// ' Estimate Negative Binomial regressioin coefficients, dispersion parameter, and random intercept variance using MCMC with a weighted least squares proposal
// '
// ' This function estimates regression parameters and ...
// '
// '
// '
// ' @param counts A numeric matrix of RNA-Seq counts (rows are genes, columns are samples)
// ' @param design_mat The fixed effects design matrix for mean response
// ' @param design_mat_re The design matrix for random intercepts
// ' @param contrast_mat A numeric matrix of linear contrasts of fixed effects to be tested.  Each row is considered to be an independent test, and each are done seperately
// ' @param prior_sd_betas Prior std. dev. in normal prior for regression coefficients
// ' @param prior_sd_betas_a Alpha parameter in inverse gamma prior for random intercept variance
// ' @param prior_sd_betas_b Beta parameter in inverse gamma prior for random intercept variance
// ' @param prior_sd_rs Prior std. dev in log-normal prior for dispersion parameters
// ' @param prior_mean_log_rs Vector of prior means for log of dispersion parameters
// ' @param rw_sd_rs Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
// ' @param n_it Number of iterations to run MCMC
// ' @param log_offset Vector of offsets on log scale
// ' @param prop_burn_in Proportion of MCMC chain to discard as burn-in when computing summaries
// ' @param starting_betas Numeric matrix of starting values for the regression coefficients.  For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
// ' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC.  In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
// '
// ' @author Brian Vestal
// '
// ' @return
// ' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
// '
// ' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_wls(arma::mat counts,
                           arma::mat design_mat,
                           arma::mat design_mat_re,
                           arma::mat contrast_mat,
                           double prior_sd_betas,
                           double prior_sd_betas_a,
                           double prior_sd_betas_b,
                           double prior_sd_rs,
                           arma::vec prior_mean_log_rs,
                           int n_it,
                           double rw_sd_rs,
                           arma::vec log_offset,
                           arma::mat starting_betas,
                           double prop_burn_in = 0.10,
                           int num_accept = 20,
                           bool return_cont = false,
                           Rcpp::StringVector beta_names = NA_STRING,
                           Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par_sum_cont_pb(counts,
                                   log_offset,
                                   starting_betas2,
                                   design_mat_tot,
                                   cont_mat_trans,
                                   prior_mean_log_rs,
                                   prior_sd_rs,
                                   rw_sd_rs,
                                   prior_sd_betas,
                                   n_beta,
                                   n_beta_re,
                                   n_sample,
                                   prior_sd_betas_a,
                                   prior_sd_betas_b,
                                   n_it,
                                   num_accept,
                                   prop_burn_in,
                                   return_cont);

  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alpha;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 3));
  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev",
                                                              "BF_exact", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;
  disp_ret = ret.tube(0, 4);
  sigma2_ret = ret.tube(0, 5);
  accepts_ret = ret.tube(0, 6);
  accepts_ret_alpha = ret.tube(0, 7);

  arma::mat geweke_ret_beta, geweke_ret_alpha, geweke_ret_sig2;
  geweke_ret_beta = ret.tube(arma::span(0, n_beta - 1), arma::span(8));
  geweke_ret_alpha = ret.tube(0, 9);
  geweke_ret_sig2 = ret.tube(0, 10);
  arma::mat ret_gwe = arma::join_horiz(geweke_ret_beta.t(), geweke_ret_alpha.t());
  ret_gwe = arma::join_horiz(ret_gwe, geweke_ret_sig2.t());

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 3));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);
    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
}



///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////






//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_sum_cont_pb2(const arma::rowvec &counts,
                                          const arma::vec &log_offset,
                                          const arma::rowvec &starting_betas,
                                          const arma::mat &design_mat,
                                          const arma::mat &contrast_mat,
                                          const double &mean_rho,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const double &n_beta,
                                          const double &n_beta_re,
                                          const double &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it,
                                          const int &num_accept,
                                          const double &prop_burn,
                                          const bool &return_cont,
                                          const double &rw_sd_sigma){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, accepts_alpha = 0, inv_errors = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat contrast_sample;
  //double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_beta + n_cont, 11, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  double beta0_prior_mean = log(arma::mean(counts));
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;
  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force_pb(betas_last,
                                                         counts,
                                                         disp_sample(i-1),
                                                         log_offset,
                                                         design_mat,
                                                         prior_sd_betas,
                                                         sigma2_sample(i-1),
                                                         n_beta,
                                                         n_beta_re,
                                                         n_sample,
                                                         accepts,
                                                         i,
                                                         num_accept,
                                                         inv_errors,
                                                         beta0_prior_mean));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);

    //b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    //sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    sigma2_sample(i) = update_sigma2_rw(sigma2_sample(i-1), rw_sd_sigma, beta_cur_re);
    i++;
  }
  if(inv_errors > 0){
    ret.fill(NA_REAL);
    accepts = -1;
    ret(0, 7) = accepts;
    return(ret);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  double var_first, var_second, mean_first, mean_second, z_g;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 4) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 5) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 6) = accepts;
  ret(0, 7) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    //ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 2) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 3) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
  }

  // Calculating Geweke p-values for regression coefficients
  arma::mat betas_nb = betas_sample.rows(burn_bound, n_it - 1);
  arma::vec disp_nb = disp_sample.rows(burn_bound, n_it - 1);
  arma::vec sigma2_nb = sigma2_sample.rows(burn_bound, n_it - 1);
  int n_row_nb = betas_nb.n_rows, n_thin = 40;
  arma::uvec idx_thin = arma::regspace<arma::uvec>(0, n_thin, n_row_nb-1);

  betas_nb = betas_nb.rows(idx_thin);
  disp_nb = arma::log(disp_nb.rows(idx_thin));
  sigma2_nb = arma::log(sigma2_nb.rows(idx_thin));
  n_row_nb = betas_nb.n_rows;

  int gub = round(n_row_nb * 0.2);
  int glb = round(n_row_nb * 0.5);

  double df_t;

  for(int kk = 0; kk < n_beta; kk++){
    var_first = arma::var(betas_nb.rows(0, gub - 1).col(kk)) / gub;
    var_second = arma::var(betas_nb.rows(glb - 1, n_row_nb - 1).col(kk)) / (n_row_nb / 2.0);
    mean_first = arma::mean(betas_nb.rows(0, gub - 1).col(kk));
    mean_second = arma::mean(betas_nb.rows(glb, n_row_nb - 1).col(kk));
    z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
    df_t = pow(var_first + var_second, 2) /
      (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
    //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
    ret(kk, 8) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);
  }

  // Calculating Geweke p-values for dispersion
  var_first = arma::var(disp_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(disp_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(disp_nb.rows(0, gub - 1));
  mean_second = arma::mean(disp_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 9) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  // Calculating Geweke p-values for RI variance
  var_first = arma::var(sigma2_nb.rows(0, gub - 1)) / gub;
  var_second = arma::var(sigma2_nb.rows(glb - 1, n_row_nb - 1)) / (n_row_nb / 2.0);
  mean_first = arma::mean(sigma2_nb.rows(0, gub - 1));
  mean_second = arma::mean(sigma2_nb.rows(glb, n_row_nb - 1));
  z_g = (mean_first - mean_second) / sqrt(var_first + var_second);
  df_t = pow(var_first + var_second, 2) /
    (pow(var_first, 2) / (gub - 1) + pow(var_second, 2) / ((n_row_nb / 2.0) - 1));
  //ret(kk, 10) = 2.0 * R::pnorm5(fabs(z_g), 0, 1, 0, 0);
  ret(0, 10) = 2.0 * R::pt(fabs(z_g), df_t, 0, 0);

  return(ret);
}

struct whole_feature_sample_struct_sum_cont_pb2 : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const int &num_accept;
  const double &prop_burn;
  const bool &return_cont;
  const double &rw_sd_sigma;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_sum_cont_pb2(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::mat &contrast_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_beta_re,
                                           const int &n_sample,
                                           const double &prior_sd_betas_a,
                                           const double &prior_sd_betas_b,
                                           const int &n_it,
                                           const int &num_accept,
                                           const double &prop_burn,
                                           const bool &return_cont,
                                           const double &rw_sd_sigma,
                                           arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), return_cont(return_cont), rw_sd_sigma(rw_sd_sigma), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_sum_cont_pb2(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      num_accept,
                      prop_burn,
                      return_cont,
                      rw_sd_sigma);
    }
  }

};

arma::cube mcmc_chain_par_sum_cont_pb2(const arma::mat &counts,
                                       const arma::vec &log_offset,
                                       const arma::mat &starting_betas,
                                       const arma::mat &design_mat,
                                       const arma::mat &contrast_mat,
                                       const arma::vec &mean_rhos,
                                       const double &prior_sd_rs,
                                       const double &rw_sd_rs,
                                       const double &prior_sd_betas,
                                       const int &n_beta,
                                       const int &n_beta_re,
                                       const int &n_sample,
                                       const double &prior_sd_betas_a,
                                       const double &prior_sd_betas_b,
                                       const int &n_it,
                                       const int &num_accept,
                                       const double &prop_burn,
                                       const bool &return_cont,
                                       const double &rw_sd_sigma){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 11, counts.n_rows, arma::fill::zeros);
  whole_feature_sample_struct_sum_cont_pb2 mcmc_inst(counts,
                                                     log_offset,
                                                     starting_betas,
                                                     design_mat,
                                                     contrast_mat,
                                                     mean_rhos,
                                                     prior_sd_rs,
                                                     rw_sd_rs,
                                                     prior_sd_betas,
                                                     n_beta,
                                                     n_beta_re,
                                                     n_sample,
                                                     prior_sd_betas_a,
                                                     prior_sd_betas_b,
                                                     n_it,
                                                     num_accept,
                                                     prop_burn,
                                                     return_cont,
                                                     rw_sd_sigma,
                                                     upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

//' Negative Binomial GLMM fit for RNA-Seq expression using MCMC (contrast version)
//'
//' Estimate Negative Binomial regressioin coefficients, dispersion parameter, and random intercept variance using MCMC with a weighted least squares proposal
//'
//' This function estimates regression parameters and ...
//'
//'
//'
//' @param counts A numeric matrix of RNA-Seq counts (rows are genes, columns are samples)
//' @param design_mat The fixed effects design matrix for mean response
//' @param design_mat_re The design matrix for random intercepts
//' @param contrast_mat A numeric matrix of linear contrasts of fixed effects to be tested.  Each row is considered to be an independent test, and each are done seperately
//' @param prior_sd_betas Prior std. dev. in normal prior for regression coefficients
//' @param prior_sd_betas_a Alpha parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b Beta parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_rs Prior std. dev in log-normal prior for dispersion parameters
//' @param prior_mean_log_rs Vector of prior means for log of dispersion parameters
//' @param rw_sd_rs Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
//' @param n_it Number of iterations to run MCMC
//' @param log_offset Vector of offsets on log scale
//' @param prop_burn_in Proportion of MCMC chain to discard as burn-in when computing summaries
//' @param starting_betas Numeric matrix of starting values for the regression coefficients.  For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
//' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC.  In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_wls2(arma::mat counts,
                            arma::mat design_mat,
                            arma::mat design_mat_re,
                            arma::mat contrast_mat,
                            double prior_sd_betas,
                            double prior_sd_betas_a,
                            double prior_sd_betas_b,
                            double prior_sd_rs,
                            arma::vec prior_mean_log_rs,
                            int n_it,
                            double rw_sd_rs,
                            double rw_sd_sigma,
                            arma::vec log_offset,
                            arma::mat starting_betas,
                            double prop_burn_in = 0.10,
                            int num_accept = 20,
                            bool return_cont = false,
                            Rcpp::StringVector beta_names = NA_STRING,
                            Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par_sum_cont_pb2(counts,
                                    log_offset,
                                    starting_betas2,
                                    design_mat_tot,
                                    cont_mat_trans,
                                    prior_mean_log_rs,
                                    prior_sd_rs,
                                    rw_sd_rs,
                                    prior_sd_betas,
                                    n_beta,
                                    n_beta_re,
                                    n_sample,
                                    prior_sd_betas_a,
                                    prior_sd_betas_b,
                                    n_it,
                                    num_accept,
                                    prop_burn_in,
                                    return_cont,
                                    rw_sd_sigma);

  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alpha;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 3));
  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev",
                                                              "BF_exact", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;
  disp_ret = ret.tube(0, 4);
  sigma2_ret = ret.tube(0, 5);
  accepts_ret = ret.tube(0, 6);
  accepts_ret_alpha = ret.tube(0, 7);

  arma::mat geweke_ret_beta, geweke_ret_alpha, geweke_ret_sig2;
  geweke_ret_beta = ret.tube(arma::span(0, n_beta - 1), arma::span(8));
  geweke_ret_alpha = ret.tube(0, 9);
  geweke_ret_sig2 = ret.tube(0, 10);
  arma::mat ret_gwe = arma::join_horiz(geweke_ret_beta.t(), geweke_ret_alpha.t());
  ret_gwe = arma::join_horiz(ret_gwe, geweke_ret_sig2.t());

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 3));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);
    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha,
                              Rcpp::Named("geweke_all") = ret_gwe);
  }
}



///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////






//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_sum_cont_pb3(const arma::rowvec &counts,
                                          const arma::vec &log_offset,
                                          const arma::rowvec &starting_betas,
                                          const arma::mat &design_mat,
                                          const arma::mat &contrast_mat,
                                          const double &mean_rho,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const double &n_beta,
                                          const double &n_beta_re,
                                          const double &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it,
                                          const int &num_accept,
                                          const double &prop_burn,
                                          const bool &return_cont,
                                          const double &rw_sd_sigma,
                                          const double &tau){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, accepts_alpha = 0, inv_errors = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat contrast_sample;
  //double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_beta + n_cont, 10, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  double beta0_prior_mean = log(arma::mean(counts));
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;
  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force_pb(betas_last,
                                                         counts,
                                                         disp_sample(i-1),
                                                         log_offset,
                                                         design_mat,
                                                         prior_sd_betas,
                                                         sigma2_sample(i-1),
                                                         n_beta,
                                                         n_beta_re,
                                                         n_sample,
                                                         accepts,
                                                         i,
                                                         num_accept,
                                                         inv_errors,
                                                         beta0_prior_mean));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);

    //b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    //sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    sigma2_sample(i) = update_sigma2_rw_hc(sigma2_sample(i-1), rw_sd_sigma, beta_cur_re, tau);
    i++;
  }
  if(inv_errors > 0){
    ret.fill(NA_REAL);
    accepts = -1;
    ret(0, 7) = accepts;
    return(ret);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 6) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 7) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 8) = accepts;
  ret(0, 9) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 3) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 5) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
  }
  return(ret);
}

struct whole_feature_sample_struct_sum_cont_pb3 : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const int &num_accept;
  const double &prop_burn;
  const bool &return_cont;
  const double &rw_sd_sigma;
  const double &tau;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_sum_cont_pb3(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::mat &contrast_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_beta_re,
                                           const int &n_sample,
                                           const double &prior_sd_betas_a,
                                           const double &prior_sd_betas_b,
                                           const int &n_it,
                                           const int &num_accept,
                                           const double &prop_burn,
                                           const bool &return_cont,
                                           const double &rw_sd_sigma,
                                           const double &tau,
                                           arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), return_cont(return_cont), rw_sd_sigma(rw_sd_sigma), tau(tau),
      upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_sum_cont_pb3(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      num_accept,
                      prop_burn,
                      return_cont,
                      rw_sd_sigma,
                      tau);
    }
  }

};

arma::cube mcmc_chain_par_sum_cont_pb3(const arma::mat &counts,
                                       const arma::vec &log_offset,
                                       const arma::mat &starting_betas,
                                       const arma::mat &design_mat,
                                       const arma::mat &contrast_mat,
                                       const arma::vec &mean_rhos,
                                       const double &prior_sd_rs,
                                       const double &rw_sd_rs,
                                       const double &prior_sd_betas,
                                       const int &n_beta,
                                       const int &n_beta_re,
                                       const int &n_sample,
                                       const double &prior_sd_betas_a,
                                       const double &prior_sd_betas_b,
                                       const int &n_it,
                                       const int &num_accept,
                                       const double &prop_burn,
                                       const bool &return_cont,
                                       const double &rw_sd_sigma,
                                       const double &tau){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 10, counts.n_rows, arma::fill::zeros);
  whole_feature_sample_struct_sum_cont_pb3 mcmc_inst(counts,
                                                     log_offset,
                                                     starting_betas,
                                                     design_mat,
                                                     contrast_mat,
                                                     mean_rhos,
                                                     prior_sd_rs,
                                                     rw_sd_rs,
                                                     prior_sd_betas,
                                                     n_beta,
                                                     n_beta_re,
                                                     n_sample,
                                                     prior_sd_betas_a,
                                                     prior_sd_betas_b,
                                                     n_it,
                                                     num_accept,
                                                     prop_burn,
                                                     return_cont,
                                                     rw_sd_sigma,
                                                     tau,
                                                     upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

//' Negative Binomial GLMM fit for RNA-Seq expression using MCMC (contrast version)
//'
//' Estimate Negative Binomial regressioin coefficients, dispersion parameter, and random intercept variance using MCMC with a weighted least squares proposal
//'
//' This function estimates regression parameters and ...
//'
//'
//'
//' @param counts A numeric matrix of RNA-Seq counts (rows are genes, columns are samples)
//' @param design_mat The fixed effects design matrix for mean response
//' @param design_mat_re The design matrix for random intercepts
//' @param contrast_mat A numeric matrix of linear contrasts of fixed effects to be tested.  Each row is considered to be an independent test, and each are done seperately
//' @param prior_sd_betas Prior std. dev. in normal prior for regression coefficients
//' @param prior_sd_betas_a Alpha parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b Beta parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_rs Prior std. dev in log-normal prior for dispersion parameters
//' @param prior_mean_log_rs Vector of prior means for log of dispersion parameters
//' @param rw_sd_rs Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
//' @param n_it Number of iterations to run MCMC
//' @param log_offset Vector of offsets on log scale
//' @param prop_burn_in Proportion of MCMC chain to discard as burn-in when computing summaries
//' @param starting_betas Numeric matrix of starting values for the regression coefficients.  For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
//' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC.  In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_wls3(arma::mat counts,
                            arma::mat design_mat,
                            arma::mat design_mat_re,
                            arma::mat contrast_mat,
                            double prior_sd_betas,
                            double prior_sd_betas_a,
                            double prior_sd_betas_b,
                            double prior_sd_rs,
                            arma::vec prior_mean_log_rs,
                            int n_it,
                            double rw_sd_rs,
                            double rw_sd_sigma,
                            arma::vec log_offset,
                            arma::mat starting_betas,
                            double prop_burn_in = 0.10,
                            double tau = 10,
                            int num_accept = 20,
                            bool return_cont = false,
                            Rcpp::StringVector beta_names = NA_STRING,
                            Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par_sum_cont_pb3(counts,
                                    log_offset,
                                    starting_betas2,
                                    design_mat_tot,
                                    cont_mat_trans,
                                    prior_mean_log_rs,
                                    prior_sd_rs,
                                    rw_sd_rs,
                                    prior_sd_betas,
                                    n_beta,
                                    n_beta_re,
                                    n_sample,
                                    prior_sd_betas_a,
                                    prior_sd_betas_b,
                                    n_it,
                                    num_accept,
                                    prop_burn_in,
                                    return_cont,
                                    rw_sd_sigma,
                                    tau);

  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alpha;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 5));
  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev", "BF_norm",
                                                              "BF_exact", "p_val_norm", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;
  disp_ret = ret.tube(0, 6);
  sigma2_ret = ret.tube(0, 7);
  accepts_ret = ret.tube(0, 8);
  accepts_ret_alpha = ret.tube(0, 9);

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 5));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);
    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha);
  }
}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////






//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_sum_cont_pb4(const arma::rowvec &counts,
                                          const arma::vec &log_offset,
                                          const arma::rowvec &starting_betas,
                                          const arma::mat &design_mat,
                                          const arma::mat &contrast_mat,
                                          const double &mean_rho,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const double &n_beta,
                                          const double &n_beta_re,
                                          const double &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it,
                                          const int &num_accept,
                                          const double &prop_burn,
                                          const bool &return_cont,
                                          const double &rw_sd_sigma,
                                          const double &tau){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, accepts_alpha = 0, inv_errors = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat contrast_sample;
  //double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_beta + n_cont, 10, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  double beta0_prior_mean = log(arma::mean(counts));
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;
  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force_pb(betas_last,
                                                         counts,
                                                         disp_sample(i-1),
                                                         log_offset,
                                                         design_mat,
                                                         prior_sd_betas,
                                                         sigma2_sample(i-1),
                                                         n_beta,
                                                         n_beta_re,
                                                         n_sample,
                                                         accepts,
                                                         i,
                                                         num_accept,
                                                         inv_errors,
                                                         beta0_prior_mean));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
    disp_sample(i) = update_rho_force_hc(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alpha);

    //b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    //sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    sigma2_sample(i) = update_sigma2_rw_hc(sigma2_sample(i-1), rw_sd_sigma, beta_cur_re, tau);
    i++;
  }
  if(inv_errors > 0){
    ret.fill(NA_REAL);
    accepts = -1;
    ret(0, 7) = accepts;
    return(ret);
  }

  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth, cp_pdf_0;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  arma::uvec idx_less;
  arma::vec pdf_vals_cp;
  if(return_cont){
    contrast_sample = betas_sample * contrast_mat;
    betas_sample = arma::join_rows(betas_sample, contrast_sample);
  }
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 6) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 7) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 8) = accepts;
  ret(0, 9) = accepts_alpha;
  for(int k = 0; k < n_beta + n_cont; k++){
    ret(k, 2) = R::dnorm4(0, ret(k, 0), ret(k, 1), 0) / R::dnorm4(0, 0, prior_sd_betas, 0);
    sd_smooth = 1.06 * ret(k, 1) * pow(n_it_double - n_burn_in, -0.20);
    pdf_vals = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), 0, sd_smooth);
    ret(k, 3) = arma::mean(pdf_vals) / R::dnorm4(0, 0, prior_sd_betas, 0);
    //ret(k, 4) = 2.0 * R::pnorm5(0, fabs(ret(k, 0)), ret(k, 1), 1, 0);
    idx_ops = arma::find((ret(k, 0) * betas_sample.rows(burn_bound, n_it - 1).col(k)) < 0);
    ret(k, 5) = (2.0 * idx_ops.n_elem) / (n_it_double - n_burn_in);
    pdf_vals_cp = arma::normpdf(betas_sample.rows(burn_bound, n_it - 1).col(k), ret(k, 0), ret(k, 1));
    cp_pdf_0 = R::dnorm4(0, ret(k, 0), ret(k, 1), 0);
    idx_less = arma::find(pdf_vals_cp < cp_pdf_0);
    ret(k, 4) = (1.0 * idx_less.n_elem) / (n_it_double - n_burn_in);
  }
  return(ret);
}

struct whole_feature_sample_struct_sum_cont_pb4 : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const int &num_accept;
  const double &prop_burn;
  const bool &return_cont;
  const double &rw_sd_sigma;
  const double &tau;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_sum_cont_pb4(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::mat &contrast_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_beta_re,
                                           const int &n_sample,
                                           const double &prior_sd_betas_a,
                                           const double &prior_sd_betas_b,
                                           const int &n_it,
                                           const int &num_accept,
                                           const double &prop_burn,
                                           const bool &return_cont,
                                           const double &rw_sd_sigma,
                                           const double &tau,
                                           arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), return_cont(return_cont), rw_sd_sigma(rw_sd_sigma), tau(tau),
      upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_sum_cont_pb4(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      num_accept,
                      prop_burn,
                      return_cont,
                      rw_sd_sigma,
                      tau);
    }
  }

};

arma::cube mcmc_chain_par_sum_cont_pb4(const arma::mat &counts,
                                       const arma::vec &log_offset,
                                       const arma::mat &starting_betas,
                                       const arma::mat &design_mat,
                                       const arma::mat &contrast_mat,
                                       const arma::vec &mean_rhos,
                                       const double &prior_sd_rs,
                                       const double &rw_sd_rs,
                                       const double &prior_sd_betas,
                                       const int &n_beta,
                                       const int &n_beta_re,
                                       const int &n_sample,
                                       const double &prior_sd_betas_a,
                                       const double &prior_sd_betas_b,
                                       const int &n_it,
                                       const int &num_accept,
                                       const double &prop_burn,
                                       const bool &return_cont,
                                       const double &rw_sd_sigma,
                                       const double &tau){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_beta + n_cont, 10, counts.n_rows, arma::fill::zeros);
  whole_feature_sample_struct_sum_cont_pb4 mcmc_inst(counts,
                                                     log_offset,
                                                     starting_betas,
                                                     design_mat,
                                                     contrast_mat,
                                                     mean_rhos,
                                                     prior_sd_rs,
                                                     rw_sd_rs,
                                                     prior_sd_betas,
                                                     n_beta,
                                                     n_beta_re,
                                                     n_sample,
                                                     prior_sd_betas_a,
                                                     prior_sd_betas_b,
                                                     n_it,
                                                     num_accept,
                                                     prop_burn,
                                                     return_cont,
                                                     rw_sd_sigma,
                                                     tau,
                                                     upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

//' Negative Binomial GLMM fit for RNA-Seq expression using MCMC (contrast version)
//'
//' Estimate Negative Binomial regressioin coefficients, dispersion parameter, and random intercept variance using MCMC with a weighted least squares proposal
//'
//' This function estimates regression parameters and ...
//'
//'
//'
//' @param counts A numeric matrix of RNA-Seq counts (rows are genes, columns are samples)
//' @param design_mat The fixed effects design matrix for mean response
//' @param design_mat_re The design matrix for random intercepts
//' @param contrast_mat A numeric matrix of linear contrasts of fixed effects to be tested.  Each row is considered to be an independent test, and each are done seperately
//' @param prior_sd_betas Prior std. dev. in normal prior for regression coefficients
//' @param prior_sd_betas_a Alpha parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b Beta parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_rs Prior std. dev in log-normal prior for dispersion parameters
//' @param prior_mean_log_rs Vector of prior means for log of dispersion parameters
//' @param rw_sd_rs Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
//' @param n_it Number of iterations to run MCMC
//' @param log_offset Vector of offsets on log scale
//' @param prop_burn_in Proportion of MCMC chain to discard as burn-in when computing summaries
//' @param starting_betas Numeric matrix of starting values for the regression coefficients.  For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
//' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC.  In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_wls4(arma::mat counts,
                            arma::mat design_mat,
                            arma::mat design_mat_re,
                            arma::mat contrast_mat,
                            double prior_sd_betas,
                            double prior_sd_betas_a,
                            double prior_sd_betas_b,
                            double prior_sd_rs,
                            arma::vec prior_mean_log_rs,
                            int n_it,
                            double rw_sd_rs,
                            double rw_sd_sigma,
                            arma::vec log_offset,
                            arma::mat starting_betas,
                            double prop_burn_in = 0.10,
                            double tau = 10,
                            int num_accept = 20,
                            bool return_cont = false,
                            Rcpp::StringVector beta_names = NA_STRING,
                            Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par_sum_cont_pb3(counts,
                                    log_offset,
                                    starting_betas2,
                                    design_mat_tot,
                                    cont_mat_trans,
                                    prior_mean_log_rs,
                                    prior_sd_rs,
                                    rw_sd_rs,
                                    prior_sd_betas,
                                    n_beta,
                                    n_beta_re,
                                    n_sample,
                                    prior_sd_betas_a,
                                    prior_sd_betas_b,
                                    n_it,
                                    num_accept,
                                    prop_burn_in,
                                    return_cont,
                                    rw_sd_sigma,
                                    tau);

  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;
  arma::vec accepts_ret_alpha;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 5));
  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev", "BF_norm",
                                                              "BF_exact", "p_val_norm", "p_val_exact");
  Rcpp::colnames(betas_ret2) = names;
  Rcpp::rownames(betas_ret2) = beta_names;
  disp_ret = ret.tube(0, 6);
  sigma2_ret = ret.tube(0, 7);
  accepts_ret = ret.tube(0, 8);
  accepts_ret_alpha = ret.tube(0, 9);

  if(return_cont){
    contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 5));
    Rcpp::NumericVector contrast_ret2;
    contrast_ret2 = Rcpp::wrap(contrast_ret);
    Rcpp::rownames(contrast_ret2) = cont_names;
    Rcpp::colnames(contrast_ret2) = names;

    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("contrast_est") = contrast_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                              Rcpp::Named("alphas_est") = disp_ret,
                              Rcpp::Named("sig2_est") = sigma2_ret,
                              Rcpp::Named("accepts_betas") = accepts_ret,
                              Rcpp::Named("accepts_alphas") = accepts_ret_alpha);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
////////////////////
////////////////////
////////////////////   Versions that return the entire chain
////////////////////
////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_sum_cont_pb_wc(const arma::rowvec &counts,
                                            const arma::vec &log_offset,
                                            const arma::rowvec &starting_betas,
                                            const arma::mat &design_mat,
                                            const arma::mat &contrast_mat,
                                            const double &mean_rho,
                                            const double &prior_sd_rs,
                                            const double &rw_sd_rs,
                                            const double &prior_sd_betas,
                                            const double &n_beta,
                                            const double &n_beta_re,
                                            const double &n_sample,
                                            const double &prior_sd_betas_a,
                                            const double &prior_sd_betas_b,
                                            const int &n_it,
                                            const int &num_accept,
                                            const double &prop_burn,
                                            const bool &return_cont){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, accepts_alphas = 0, inv_errors = 0, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::mat contrast_sample;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_it, n_beta + 2, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  double beta0_prior_mean = log(arma::mean(counts));
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;
  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force_pb(betas_last,
                                                         counts,
                                                         disp_sample(i-1),
                                                         log_offset,
                                                         design_mat,
                                                         prior_sd_betas,
                                                         sigma2_sample(i-1),
                                                         n_beta,
                                                         n_beta_re,
                                                         n_sample,
                                                         accepts,
                                                         i,
                                                         num_accept,
                                                         inv_errors,
                                                         beta0_prior_mean));
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
    disp_sample(i) = update_rho_force(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta_tot,
                n_sample,
                i,
                num_accept,
                accepts_alphas);

    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    i++;
  }
  if(inv_errors > 0){
    ret.fill(NA_REAL);
    accepts = -1;
    ret(0, 7) = accepts;
    return(ret);
  }

  ret.cols(0, n_beta - 1) = betas_sample;
  ret.col(n_beta) = disp_sample;
  ret.col(n_beta + 1) = sigma2_sample;

  return(ret);
}

struct whole_feature_sample_struct_sum_cont_pb_wc : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
  const arma::mat &contrast_mat;
  const arma::vec &mean_rhos;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;
  const int &num_accept;
  const double &prop_burn;
  const bool &return_cont;

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_sum_cont_pb_wc(const arma::mat &counts,
                                          const arma::vec &log_offset,
                                          const arma::mat &starting_betas,
                                          const arma::mat &design_mat,
                                          const arma::mat &contrast_mat,
                                          const arma::vec &mean_rhos,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const int &n_beta,
                                          const int &n_beta_re,
                                          const int &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it,
                                          const int &num_accept,
                                          const double &prop_burn,
                                          const bool &return_cont,
                                          arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), return_cont(return_cont), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_sum_cont_pb_wc(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      contrast_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
                      prior_sd_betas,
                      n_beta,
                      n_beta_re,
                      n_sample,
                      prior_sd_betas_a,
                      prior_sd_betas_b,
                      n_it,
                      num_accept,
                      prop_burn,
                      return_cont);
    }
  }

};

arma::cube mcmc_chain_par_sum_cont_pb_wc(const arma::mat &counts,
                                      const arma::vec &log_offset,
                                      const arma::mat &starting_betas,
                                      const arma::mat &design_mat,
                                      const arma::mat &contrast_mat,
                                      const arma::vec &mean_rhos,
                                      const double &prior_sd_rs,
                                      const double &rw_sd_rs,
                                      const double &prior_sd_betas,
                                      const int &n_beta,
                                      const int &n_beta_re,
                                      const int &n_sample,
                                      const double &prior_sd_betas_a,
                                      const double &prior_sd_betas_b,
                                      const int &n_it,
                                      const int &num_accept,
                                      const double &prop_burn,
                                      const bool &return_cont){
  int n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_cols;
  }
  arma::cube upd_param(n_it, n_beta + 2, counts.n_rows, arma::fill::zeros);
  whole_feature_sample_struct_sum_cont_pb_wc mcmc_inst(counts,
                                                    log_offset,
                                                    starting_betas,
                                                    design_mat,
                                                    contrast_mat,
                                                    mean_rhos,
                                                    prior_sd_rs,
                                                    rw_sd_rs,
                                                    prior_sd_betas,
                                                    n_beta,
                                                    n_beta_re,
                                                    n_sample,
                                                    prior_sd_betas_a,
                                                    prior_sd_betas_b,
                                                    n_it,
                                                    num_accept,
                                                    prop_burn,
                                                    return_cont,
                                                    upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

//' Negative Binomial GLMM fit for RNA-Seq expression using MCMC (contrast version)
//'
//' Estimate Negative Binomial regressioin coefficients, dispersion parameter, and random intercept variance using MCMC with a weighted least squares proposal
//'
//' This function estimates regression parameters and ...
//'
//'
//'
//' @param counts A numeric matrix of RNA-Seq counts (rows are genes, columns are samples)
//' @param design_mat The fixed effects design matrix for mean response
//' @param design_mat_re The design matrix for random intercepts
//' @param contrast_mat A numeric matrix of linear contrasts of fixed effects to be tested.  Each row is considered to be an independent test, and each are done seperately
//' @param prior_sd_betas Prior std. dev. in normal prior for regression coefficients
//' @param prior_sd_betas_a Alpha parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b Beta parameter in inverse gamma prior for random intercept variance
//' @param prior_sd_rs Prior std. dev in log-normal prior for dispersion parameters
//' @param prior_mean_log_rs Vector of prior means for log of dispersion parameters
//' @param rw_sd_rs Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
//' @param n_it Number of iterations to run MCMC
//' @param log_offset Vector of offsets on log scale
//' @param prop_burn_in Proportion of MCMC chain to discard as burn-in when computing summaries
//' @param starting_betas Numeric matrix of starting values for the regression coefficients.  For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
//' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC.  In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
////' @export
// [[Rcpp::export]]

Rcpp::List nbglmm_mcmc_wls_wc(arma::mat counts,
                           arma::mat design_mat,
                           arma::mat design_mat_re,
                           arma::mat contrast_mat,
                           double prior_sd_betas,
                           double prior_sd_betas_a,
                           double prior_sd_betas_b,
                           double prior_sd_rs,
                           arma::vec prior_mean_log_rs,
                           int n_it,
                           double rw_sd_rs,
                           arma::vec log_offset,
                           arma::mat starting_betas,
                           double prop_burn_in = 0.10,
                           int num_accept = 20,
                           bool return_cont = false,
                           Rcpp::StringVector beta_names = NA_STRING,
                           Rcpp::StringVector cont_names = NA_STRING){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = 0;
  if(return_cont){
    n_cont = contrast_mat.n_rows;
  }
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re), cont_mat_trans = contrast_mat.t();
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par_sum_cont_pb_wc(counts,
                                   log_offset,
                                   starting_betas2,
                                   design_mat_tot,
                                   cont_mat_trans,
                                   prior_mean_log_rs,
                                   prior_sd_rs,
                                   rw_sd_rs,
                                   prior_sd_betas,
                                   n_beta,
                                   n_beta_re,
                                   n_sample,
                                   prior_sd_betas_a,
                                   prior_sd_betas_b,
                                   n_it,
                                   num_accept,
                                   prop_burn_in,
                                   return_cont);

  // arma::cube betas_ret;
  //arma::cube contrast_ret;
  // arma::mat disp_ret, sigma2_ret;
  //arma::vec accepts_ret;
  //arma::vec accepts_ret_alpha;

  // betas_ret = ret.cols(0, n_beta - 1);
  // disp_ret = ret.col(n_beta);
  // sigma2_ret = ret.col(n_beta+1);


  if(return_cont){

    // return Rcpp::List::create(Rcpp::Named("ret") = betas_ret,
    //                           Rcpp::Named("alphas_est") = disp_ret,
    //                           Rcpp::Named("sig2_est") = sigma2_ret);
    return Rcpp::List::create(Rcpp::Named("ret") = ret);

  }
  else{
    // return Rcpp::List::create(Rcpp::Named("ret") = betas_ret,
    //                           Rcpp::Named("alphas_est") = disp_ret,
    //                           Rcpp::Named("sig2_est") = sigma2_ret);
    return Rcpp::List::create(Rcpp::Named("ret") = ret);
  }
}

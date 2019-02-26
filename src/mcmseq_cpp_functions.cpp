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
    }
  }

  // Do the transformation
  ans = ans * csigma;
  ans.each_row() += mu_tmp.t(); // Add mu to each row in transformed ans

  return ans;
}

////////////////////////////////////////////////////////////////////////////////
/*
 * Basic functions for doing NBGLM MCMC parameter updates
 */
////////////////////////////////////////////////////////////////////////////////

//    Function to update all regression parameters for a single feature using a
//    random walk proposal

arma::vec update_betas(const arma::rowvec &beta_cur,
                       const arma::rowvec &counts,
                       const double &alpha_cur,
                       const arma::vec &log_offset,
                       const arma::mat &design_mat,
                       const double &prior_sd_betas,
                       const double &rw_sd_betas,
                       const int &n_beta,
                       const int &n_sample){
  int k;
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  mean_cur = arma::exp(design_mat * beta_cur_tmp + log_offset);
  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  for(k = 0; k < n_beta; k++){
    beta_prop = beta_cur_tmp;
    beta_prop(k) += R::rnorm(0, rw_sd_betas);
    mean_prop = arma::exp(design_mat * beta_prop + log_offset);
    ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
    mh_cur = ll_cur + R::dnorm4(beta_cur(k), 0.0, prior_sd_betas, 1);
    mh_prop = ll_prop + R::dnorm4(beta_prop(k), 0.0, prior_sd_betas, 1);
    if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
      ll_cur = ll_prop;
      beta_cur_tmp = beta_prop;
      mean_cur = mean_prop;
    }
  }
  return beta_cur_tmp;
}

//    Function to update all regression parameters for a single feature using a
//    Wieghted Least Squares (WLS) proposal

arma::vec update_betas_wls(const arma::rowvec &beta_cur,
                           const arma::rowvec &counts,
                           const double &alpha_cur,
                           const arma::vec &log_offset,
                           const arma::mat &design_mat,
                           const double &prior_sd_betas,
                           const int &n_beta,
                           const int &n_sample,
                           const arma::vec &R_mat_diag,
                           int &accept_rec){
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta), mean_wls_prop(n_beta);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta, n_beta);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta, n_beta), cov_mat_prop(n_beta, n_beta);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  arma::vec prior_mean_betas(n_beta);
  prior_mean_betas.zeros();

  eta = design_mat * beta_cur_tmp + log_offset;
  mean_cur = arma::exp(eta);

  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;

  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();

  R_mat.zeros();
  R_mat.diag() = R_mat_diag;

  cov_mat_cur = arma::inv(R_mat.i() + design_mat.t() * W_mat * design_mat);

  mean_wls_cur = cov_mat_cur * (design_mat.t() * W_mat * y_tilde);


  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  eta_prop = design_mat * beta_prop + log_offset;
  mean_prop = arma::exp(eta_prop);

  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);
  W_mat_prop = W_mat_prop.i();

  cov_mat_prop = arma::inv(R_mat.i() + design_mat.t() * W_mat_prop * design_mat);

  mean_wls_prop = cov_mat_prop * (design_mat.t() * W_mat_prop * y_tilde_prop);

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));

  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

//    Function to update all regression parameters for a single feature using a
//    Wieghted Least Squares (WLS) proposal with safety checks for inverses

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
                  const int &n_sample){
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
                        const int &num_accept){
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
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}


////////////////////////////////////////////////////////////////////////////////
/*
 * Declaration of structures used in RcppParallel to update features in parallel
 */
////////////////////////////////////////////////////////////////////////////////

//  Structure for parallel updating of regression parameters using random walk

struct upd_betas_struct : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const double &rw_sd_betas;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_struct(const arma::mat &beta_cur,
                   const arma::mat &counts,
                   const arma::vec &alpha_cur,
                   const arma::vec &log_offset,
                   const arma::mat &design_mat,
                   const double &prior_sd_betas,
                   const double &rw_sd_betas,
                   const int &n_beta,
                   const int &n_sample,
                   arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), rw_sd_betas(rw_sd_betas), n_beta(n_beta), n_sample(n_sample),
      upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset, design_mat,
                                     prior_sd_betas, rw_sd_betas, n_beta, n_sample));
    }
  }

};



//  Structure for parallel updating of NB dispersions using random walk and log-normal prior

struct upd_rhos_struct : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &mean_alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::vec &upd_rhos;

  // constructors
  upd_rhos_struct(const arma::mat &beta_cur,
                  const arma::mat &counts,
                  const arma::vec &alpha_cur,
                  const arma::vec &mean_alpha_cur,
                  const arma::vec &log_offset,
                  const arma::mat &design_mat,
                  const double &prior_sd_rs,
                  const double &rw_sd_rs,
                  const int &n_beta,
                  const int &n_sample,
                  arma::vec &upd_rhos)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), mean_alpha_cur(mean_alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), n_beta(n_beta), n_sample(n_sample), upd_rhos(upd_rhos) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_rhos(i) = update_rho(beta_cur.row(i),
               counts.row(i),
               alpha_cur(i),
               mean_alpha_cur(i),
               log_offset,
               design_mat,
               prior_sd_rs,
               rw_sd_rs,
               n_beta,
               n_sample);
    }
  }

};

//  Structure for parallel updating of NB dispersions using random walk and log-normal prior

struct upd_rhos_struct_force : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &mean_alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const int &n_beta;
  const int &n_sample;
  const int &it_num;
  const int &num_accept;

  // updates accumulated so far
  arma::vec &upd_rhos;

  // constructors
  upd_rhos_struct_force(const arma::mat &beta_cur,
                        const arma::mat &counts,
                        const arma::vec &alpha_cur,
                        const arma::vec &mean_alpha_cur,
                        const arma::vec &log_offset,
                        const arma::mat &design_mat,
                        const double &prior_sd_rs,
                        const double &rw_sd_rs,
                        const int &n_beta,
                        const int &n_sample,
                        const int &it_num,
                        const int &num_accept,
                        arma::vec &upd_rhos)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), mean_alpha_cur(mean_alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), n_beta(n_beta), n_sample(n_sample), it_num(it_num), num_accept(num_accept), upd_rhos(upd_rhos) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_rhos(i) = update_rho_force(beta_cur.row(i),
               counts.row(i),
               alpha_cur(i),
               mean_alpha_cur(i),
               log_offset,
               design_mat,
               prior_sd_rs,
               rw_sd_rs,
               n_beta,
               n_sample,
               it_num,
               num_accept);
    }
  }

};




////////////////////////////////////////////////////////////////////////////////
/*
 * Parallelized functions for doing NBGLM MCMC parameter updates using functions
 * and structures defined in previous 2 sections with parallelFor from RcppParallel
 */
////////////////////////////////////////////////////////////////////////////////

//  Parallel updating of regression coeficients using random walk proposal

arma::mat para_update_betas(const arma::mat &beta_cur,
                            const arma::mat &counts,
                            const arma::vec &alpha_cur,
                            const arma::vec &log_offset,
                            const arma::mat &design_mat,
                            const double &prior_sd_betas,
                            const double &rw_sd_betas,
                            const int &n_beta,
                            const int &n_sample,
                            const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_struct upd_betas_inst(beta_cur,
                                  counts,
                                  alpha_cur,
                                  log_offset,
                                  design_mat,
                                  prior_sd_betas,
                                  rw_sd_betas,
                                  n_beta,
                                  n_sample,
                                  upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}


//  Parallel updating of NB dispersions using random walk proposal
arma::vec para_update_rhos(const arma::mat &beta_cur,
                           const arma::mat &counts,
                           const arma::vec &alpha_cur,
                           const arma::vec &mean_alpha_cur,
                           const arma::vec &log_offset,
                           const arma::mat &design_mat,
                           const double &prior_sd_rs,
                           const double &rw_sd_rs,
                           const int &n_beta,
                           const int &n_sample,
                           const int &grain_size){
  arma::vec upd_rhos(counts.n_rows);

  upd_rhos_struct upd_rhos_inst(beta_cur,
                                counts,
                                alpha_cur,
                                mean_alpha_cur,
                                log_offset,
                                design_mat,
                                prior_sd_rs,
                                rw_sd_rs,
                                n_beta,
                                n_sample,
                                upd_rhos);
  parallelFor(0, counts.n_rows, upd_rhos_inst, grain_size);
  return(upd_rhos);
}

//  Parallel updating of NB dispersions using random walk proposal
arma::vec para_update_rhos_force(const arma::mat &beta_cur,
                                 const arma::mat &counts,
                                 const arma::vec &alpha_cur,
                                 const arma::vec &mean_alpha_cur,
                                 const arma::vec &log_offset,
                                 const arma::mat &design_mat,
                                 const double &prior_sd_rs,
                                 const double &rw_sd_rs,
                                 const int &n_beta,
                                 const int &n_sample,
                                 const int &it_num,
                                 const int &num_accept,
                                 const int &grain_size){
  arma::vec upd_rhos(counts.n_rows);

  upd_rhos_struct_force upd_rhos_inst(beta_cur,
                                      counts,
                                      alpha_cur,
                                      mean_alpha_cur,
                                      log_offset,
                                      design_mat,
                                      prior_sd_rs,
                                      rw_sd_rs,
                                      n_beta,
                                      n_sample,
                                      it_num,
                                      num_accept,
                                      upd_rhos);
  parallelFor(0, counts.n_rows, upd_rhos_inst, grain_size);
  return(upd_rhos);
}



////////////////////////////////////////////////////////////////////////////////
/*
 * Wrapper functions to do MCMC for whole data set that are exported to R
 */
////////////////////////////////////////////////////////////////////////////////


//' Negative Binomial GLM MCMC (title)
//'
//' Run an MCMC for the Negative Binomial (short description, one or two sentences)
//'
//' This is where you write details on the function...
//'
//' more details....
//'
//' @param counts a matrix of counts
//' @param design_mat designe matrix for mean response
//' @param prior_sd_betas prior std. dev. for regression coefficients
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_betas random walk std. dev. for proposing regression coefficients
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters and a matrix of dispersion values
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc(arma::mat counts,
                      arma::mat design_mat,
                      double prior_sd_betas,
                      double prior_sd_rs,
                      arma::vec prior_mean_log_rs,
                      int n_it,
                      double rw_sd_betas,
                      double rw_sd_rs,
                      arma::vec log_offset,
                      int grain_size = 1){
  int i, n_beta = design_mat.n_cols, n_feature = counts.n_rows, n_sample = counts.n_cols;
  arma::cube betas(n_feature, n_beta, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta);
  arma::vec beta_cur(n_feature), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  betas.zeros();
  rhos.row(0) = arma::trans(arma::exp(prior_mean_log_rs));
  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat, prior_sd_betas, rw_sd_betas, n_beta, n_sample, grain_size);
    rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat, prior_sd_rs, rw_sd_rs, n_beta, n_sample, grain_size));

  }

  // Return list with posterior samples
  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                            Rcpp::Named("alphas_sample") = rhos);
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//  In this section we have similar functions for updating a NBGLMM
//  with arbitrary fixed effects and a random intercept

////////////////////////////////////////////////////////////////////////////////
/*
 * Basic functions for doing NBGLMM MCMC parameter updates
 */
////////////////////////////////////////////////////////////////////////////////

//  Function for updating fixed effects regression coefficients using a
//  random walk proposal

arma::vec update_betas_fe(const arma::rowvec &beta_cur,
                          const arma::rowvec &counts,
                          const arma::rowvec &beta_re,
                          const double &alpha_cur,
                          const arma::vec &log_offset,
                          const arma::mat &design_mat,
                          const arma::mat &design_mat_re,
                          const double &prior_sd_betas,
                          const double &rw_sd_betas,
                          const int &n_beta,
                          const int &n_sample){
  int k;
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  arma::vec re_mean(n_sample);
  arma::uvec idx_sub;

  re_mean = design_mat_re * beta_re.t();
  mean_cur = arma::exp(design_mat * beta_cur_tmp + log_offset + re_mean);
  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));

  for(k = 0; k < n_beta; k++){
    beta_prop = beta_cur_tmp;
    beta_prop(k) += R::rnorm(0, rw_sd_betas);
    mean_prop = arma::exp(design_mat * beta_prop + log_offset + re_mean);
    ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
    mh_cur = ll_cur + R::dnorm4(beta_cur(k), 0.0, prior_sd_betas, 1);
    mh_prop = ll_prop + R::dnorm4(beta_prop(k), 0.0, prior_sd_betas, 1);
    if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
      ll_cur = ll_prop;
      beta_cur_tmp = beta_prop;
      mean_cur = mean_prop;
    }
  }
  return beta_cur_tmp;
}

//  Function for updating random intercepts using a random walk proposal

arma::vec update_betas_re(const arma::rowvec &beta_cur,
                          const arma::rowvec &counts,
                          const arma::rowvec &beta_fe,
                          const double &alpha_cur,
                          const arma::vec &log_offset,
                          const arma::mat &design_mat,
                          const arma::mat &design_mat_re,
                          const double &prior_sd_betas,
                          const double &rw_sd_betas,
                          const int &n_beta,
                          const int &n_sample){
  int k;
  arma::vec beta_prop(n_beta), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  arma::vec fe_mean(n_sample);
  arma::uvec idx_sub;

  fe_mean = design_mat * beta_fe.t();
  mean_cur = arma::exp(design_mat_re * beta_cur_tmp + log_offset + fe_mean);

  for(k = 0; k < n_beta; k++){
    idx_sub = find(design_mat_re.col(k));
    ll_cur = arma::sum(arma::trans(counts(idx_sub)) * arma::log(mean_cur(idx_sub)) - (arma::trans(counts(idx_sub)) + rho_cur) * arma::log(1.0 + mean_cur(idx_sub) * alpha_cur));
    beta_prop = beta_cur_tmp;
    beta_prop(k) += R::rnorm(0, rw_sd_betas);
    mean_prop = arma::exp(design_mat_re * beta_prop + log_offset + fe_mean);
    ll_prop = arma::sum(arma::trans(counts(idx_sub)) * arma::log(mean_prop(idx_sub)) - (arma::trans(counts(idx_sub)) + rho_cur) * arma::log(1.0 + mean_prop(idx_sub) * alpha_cur));
    mh_cur = ll_cur + R::dnorm4(beta_cur(k), 0.0, prior_sd_betas, 1);
    mh_prop = ll_prop + R::dnorm4(beta_prop(k), 0.0, prior_sd_betas, 1);
    if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
      beta_cur_tmp = beta_prop;
      mean_cur = mean_prop;
    }
  }
  return beta_cur_tmp;
}

//  Function for updating NB dispersion using random walk proposal
//  (only used with random walk version of NBGLMM, WLS version uses
//  update function from NBGLM sampler)

double update_rho_mm(const arma::rowvec &beta_cur,
                     const arma::rowvec &counts,
                     const arma::rowvec &beta_re,
                     const double &alpha_cur,
                     const double &mean_alpha_cur,
                     const arma::vec &log_offset,
                     const arma::mat &design_mat,
                     const arma::mat &design_mat_re,
                     const double &prior_sd_rs,
                     const double &rw_sd_rs,
                     const int &n_beta,
                     const int &n_sample){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;

  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset + design_mat_re * beta_re.t());
  //mean_cur = arma::exp(design_mat * beta_cur.t() + design_mat_re * beta_re.t());

  alpha_prop = exp(log(alpha_cur) + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;
  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));
  mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);
  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}




//  Function for simultaneously updating fixed effects and random intercepts
//  using a WLS proposal (slightly different than NBGLM version) with forced
//  accepts for the first 50 iterations

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


////////////////////////////////////////////////////////////////////////////////
/*
 * Declaration of structures used in RcppParallel to update features in parallel
 */
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/*
 * Wrapper functions to do MCMC for whole data set that are exported to R
 */
////////////////////////////////////////////////////////////////////////////////




//  Function for simultaneously updating fixed effects and random intercepts
//  using a random walk proposal with forced accepts for the first 0 iterations

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
                                 const int &it_num){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec eta(n_sample), eta_prop(n_sample);
  arma::mat R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re);

  R_mat_diag.zeros();
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;
  prior_mean_betas.zeros();
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

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num <= 50){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}






///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/////////////////////////    GLMM RW Version      /////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm_rw2(const arma::rowvec &counts,
                                 const arma::vec &log_offset,
                                 const arma::rowvec &starting_betas,
                                 const arma::mat &design_mat,
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
                                 const int n_it){
  int n_beta_tot = n_beta + n_beta_re, accepts = 0, i, num_accept = 0;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_it, n_beta + 3);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);

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
                                                   i));
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
                num_accept);
    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
  }

  ret.cols(0, n_beta - 1) = betas_sample;
  ret.col(n_beta) = disp_sample;
  ret.col(n_beta + 1) = sigma2_sample;
  ret(0, n_beta + 2) = accepts;
  return(ret);
}


struct whole_feature_sample_rw_struct2 : public Worker
{
  // source objects
  const arma::mat &counts;
  const arma::vec &log_offset;
  const arma::mat &starting_betas;
  const arma::mat &design_mat;
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

  arma::cube &upd_param;

  // constructors
  whole_feature_sample_rw_struct2(const arma::mat &counts,
                                  const arma::vec &log_offset,
                                  const arma::mat &starting_betas,
                                  const arma::mat &design_mat,
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
                                  arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      rw_var_betas(rw_var_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it),
      upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm_rw2(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
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
                      n_it);
    }
  }

};

arma::cube mcmc_chain_rw_par2(const arma::mat &counts,
                              const arma::vec &log_offset,
                              const arma::mat &starting_betas,
                              const arma::mat &design_mat,
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
                              const int &n_it){
  arma::cube upd_param(n_it, n_beta + 3, counts.n_rows, arma::fill::zeros);

  whole_feature_sample_rw_struct2 mcmc_inst(counts,
                                            log_offset,
                                            starting_betas,
                                            design_mat,
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
                                            upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}


//' Negative Binomial GLMM MCMC Random Walk (full parallel chians)
//'
//' Run an MCMC for the Negative Binomial mixed model (short description, one or two sentences)
//'
//' This is where you write details on the function...
//'
//' more details....
//'
//' @param counts a matrix of counts
//' @param design_mat design matrix for mean response
//' @param design_mat_re design matrix for random intercepts
//' @param prior_sd_betas prior std. dev. for regression coefficients
//' @param rw_sd_betas random walk std. dev. for proposing beta values
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random walk std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_rw2(arma::mat counts,
                                 arma::mat design_mat,
                                 arma::mat design_mat_re,
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
                                 int grain_size = 1){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  double rw_var_betas = pow(rw_sd_betas, 2);
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_rw_par2(counts,
                           log_offset,
                           starting_betas2,
                           design_mat_tot,
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
                           n_it);
  arma::cube betas_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;

  betas_ret = ret.tube(arma::span(), arma::span(0, n_beta-1));
  disp_ret = ret.tube(arma::span(), arma::span(n_beta, n_beta));
  sigma2_ret = ret.tube(arma::span(), arma::span(n_beta+1, n_beta+1));
  accepts_ret = ret.tube(0, n_beta+2);

  Rcpp::NumericVector betas_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);

  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev", "BF_norm",
                                                              "BF_exact", "p_val_norm", "p_val_exact");

  Rcpp::colnames(betas_ret2) = names;

  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_ret2,
                            Rcpp::Named("alphas_sample") = disp_ret,
                            Rcpp::Named("sigma2_sample") = sigma2_ret,
                            Rcpp::Named("accepts") = accepts_ret);
}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
//////////////     GLM Summary Version (with Contrasts)      ////////////
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
                                     const double &VIF,
                                     const double &prop_burn){
  int i = 1, accepts = 0, inv_errors = 0, n_cont = contrast_mat.n_cols;
  //arma::mat ret(n_it, n_beta + 3, arma::fill::zeros);
  arma::mat ret(n_beta + n_cont, 8, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta), contrast_sample(n_it, n_cont);
  arma::rowvec betas_cur(n_beta), betas_last(n_beta);
  arma::vec disp_sample(n_it);
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec R_mat_diag(n_beta);
  R_mat_diag.fill(prior_var_betas);

  betas_sample.row(0) = starting_betas;
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  betas_cur = starting_betas;
  betas_last = starting_betas;

  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_safe(betas_last,
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
                                                  VIF));
    betas_last = betas_cur;
    betas_sample.row(i) = betas_cur;

    disp_sample(i) = update_rho(betas_cur,
                counts,
                disp_sample(i-1),
                mean_rho,
                log_offset,
                design_mat,
                prior_sd_rs,
                rw_sd_rs,
                n_beta,
                n_sample);
    i++;
  }
  if(inv_errors > 0){
    betas_sample.fill(NA_REAL);
    disp_sample.fill(NA_REAL);
    accepts = -1;
  }
  contrast_sample = betas_sample * contrast_mat;
  betas_sample = arma::join_rows(betas_sample, contrast_sample);
  int burn_bound = round(n_it * prop_burn);
  double n_it_double = n_it, n_burn_in = n_it_double * prop_burn, sd_smooth;
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 6) = arma::mean(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 7) = accepts;
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
  const double &VIF;
  const double &prop_burn;

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
                                           const double &VIF,
                                           const double &prop_burn,
                                           arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_sample(n_sample), n_it(n_it), VIF(VIF),
      prop_burn(prop_burn), upd_param(upd_param){}

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
                      VIF,
                      prop_burn);
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
                                       const double &VIF,
                                       const double &prop_burn){
  int n_cont = contrast_mat.n_cols;
  arma::cube upd_param(n_beta + n_cont, 8, counts.n_rows, arma::fill::zeros);

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
                                                     VIF,
                                                     prop_burn,
                                                     upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  return(upd_param);
}

//' Negative Binomial GLM MCMC WLS (full parallel chians)
//'
//' Run an MCMC for the Negative Binomial mixed model (short description, one or two sentences)
//'
//' This is where you write details on the function...
//'
//' more details....
//'
//' @param counts a matrix of counts
//' @param design_mat design matrix for mean response
//' @param contrast_mat contrast matrix (each row is a contrast of regression parameters to be tested)
//' @param prior_sd_betas prior std. dev. for regression coefficients
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, and a matrix of dispersion values
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc_fp_sum_cont(arma::mat counts,
                                  arma::mat design_mat,
                                  arma::mat contrast_mat,
                                  double prior_sd_betas,
                                  double prior_sd_rs,
                                  arma::vec prior_mean_log_rs,
                                  int n_it,
                                  double rw_sd_rs,
                                  arma::vec log_offset,
                                  arma::mat starting_betas,
                                  int grain_size = 1,
                                  double burn_in_prop = .1,
                                  double VIF = 1){

  int n_beta = design_mat.n_cols, n_sample = counts.n_cols, n_gene = counts.n_rows, n_cont = contrast_mat.n_rows;
  arma::cube ret(n_beta + n_cont, 8, n_gene);
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
                                    VIF,
                                    burn_in_prop);


  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret;
  arma::vec accepts_ret;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 5));
  contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 5));
  disp_ret = ret.tube(0, 6);
  accepts_ret = ret.tube(0, 7);
  //inv_errors_ret = ret.tube(0, n_beta+2);

  Rcpp::NumericVector betas_ret2;
  Rcpp::NumericVector contrast_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  contrast_ret2 = Rcpp::wrap(contrast_ret);

  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev", "BF_norm",
                                                              "BF_exact", "p_val_norm", "p_val_exact");

  Rcpp::colnames(betas_ret2) = names;
  Rcpp::colnames(contrast_ret2) = names;

  return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                            Rcpp::Named("contrast_est") = contrast_ret2,
                            Rcpp::Named("alphas_est") = disp_ret,
                            Rcpp::Named("accepts") = accepts_ret);

}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/////     NBGLMM Summary version (with Contrasts)     /////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////



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
                                         const double &prop_burn){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, inv_errors = 0, n_cont = contrast_mat.n_cols;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_beta + n_cont, 9, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta), contrast_sample(n_it, n_cont);
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
                num_accept);

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
  arma::uvec idx_ops;
  arma::vec pdf_vals;
  contrast_sample = betas_sample * contrast_mat;
  betas_sample = arma::join_rows(betas_sample, contrast_sample);
  ret.col(0) = arma::trans(arma::median(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret.col(1) = arma::trans(arma::stddev(betas_sample.rows(burn_bound, n_it - 1), 0));
  ret(0, 6) = arma::median(disp_sample.rows(burn_bound, n_it - 1));
  ret(0, 7) = arma::median(sigma2_sample.rows(burn_bound, n_it - 1));
  ret(0, 8) = accepts;
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
                                          arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      contrast_mat(contrast_mat), mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), num_accept(num_accept),
      prop_burn(prop_burn), upd_param(upd_param){}

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
                      prop_burn);
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
                                      const double &prop_burn){
  int n_cont = contrast_mat.n_cols;
  arma::cube upd_param(n_beta + n_cont, 9, counts.n_rows, arma::fill::zeros);

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
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
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

Rcpp::List nbmm_mcmc_sampler_wls_force_fp_sum_cont_pb(arma::mat counts,
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
                                                      int grain_size = 1,
                                                      int num_accept = 20){

  arma::cube ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols, n_cont = contrast_mat.n_rows;
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
                                   prop_burn_in);

  arma::cube betas_ret;
  arma::cube contrast_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret;

  betas_ret = ret.tube(arma::span(0, n_beta - 1), arma::span(0, 5));
  contrast_ret = ret.tube(arma::span(n_beta, n_beta + n_cont - 1), arma::span(0, 5));

  Rcpp::NumericVector betas_ret2;
  Rcpp::NumericVector contrast_ret2;
  betas_ret2 = Rcpp::wrap(betas_ret);
  contrast_ret2 = Rcpp::wrap(contrast_ret);

  Rcpp::CharacterVector names = Rcpp::CharacterVector::create("median", "std_dev", "BF_norm",
                                                              "BF_exact", "p_val_norm", "p_val_exact");

  Rcpp::colnames(betas_ret2) = names;
  Rcpp::colnames(contrast_ret2) = names;

  disp_ret = ret.tube(0, 6);
  sigma2_ret = ret.tube(0, 7);
  accepts_ret = ret.tube(0, 8);
  //inv_errors_ret = ret.tube(0, n_beta+2);

  return Rcpp::List::create(Rcpp::Named("betas_est") = betas_ret2,
                            Rcpp::Named("contrast_est") = contrast_ret2,
                            Rcpp::Named("alphas_est") = disp_ret,
                            Rcpp::Named("sig2_est") = sigma2_ret,
                            Rcpp::Named("accepts") = accepts_ret);
}


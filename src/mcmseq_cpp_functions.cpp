/*
 * Single File that has all NBGLM and NBGLMM MCMC fitting functions and
 * all other helper functions
 */
//#define ARMA_DONT_PRINT_ERRORS
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
  int xdim = x.n_cols;
  arma::mat rooti;
  arma::vec z;
  double out, tmp;
  //double out_win;
  double rootisum, constants;
  constants = -(static_cast<double>(xdim)/2.0) * log2pi;

  //rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  arma::mat sigma_copy = sigma;

  // Do the Cholesky decomposition
  //const arma::mat csigma = arma::chol(sigma);
  arma::mat csigma;
  bool success = false;

  while(success == false)
  {
    success = chol(csigma, sigma_copy, "upper");

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Chol failed" << std::endl;
    }
  }
  rooti = arma::trans(arma::inv(trimatu(csigma)));
  rootisum = arma::sum(log(rooti.diag()));
  z = rooti * arma::trans(x.t() - mean.t()) ;
  tmp = constants - 0.5 * arma::sum(z%z) + rootisum;
  //out_win = std::max(-100.0, tmp);
  //out = exp(out_win);
  out = exp(tmp);
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
  //const arma::mat csigma = arma::chol(sigma);
  arma::mat csigma;
  bool success = false;

  while(success == false)
  {
    success = chol(csigma, sigma_copy);

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Chol failed" << std::endl;
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
  //const arma::mat csigma = arma::chol(sigma);
  arma::mat csigma;
  bool success = false;

  while(success == false)
  {
    success = chol(csigma, sigma_copy);

    if(success == false)
    {
      sigma_copy += arma::eye(sigma.n_rows,sigma.n_rows) * 1e-4;
      //Rcpp::Rcout << "Chol failed" << std::endl;
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
//    a random walk proposal and log-normal prior (forced non-accepts for 50 it)

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
                        const int &it_num){
  arma::vec mean_cur(n_sample);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_prop, rho_cur = 1.0 / alpha_cur, alpha_prop;

  mean_cur = arma::exp(design_mat * beta_cur.t() + log_offset);

  alpha_prop = exp(log(alpha_cur) + R::rnorm(0, rw_sd_rs));
  rho_prop = 1.0 / alpha_prop;

  ll_cur = arma::sum(arma::lgamma(rho_cur + counts)) - arma::sum((counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)) - n_sample * lgamma(rho_cur) + arma::sum(counts * log(alpha_cur));
  ll_prop = arma::sum(arma::lgamma(rho_prop + counts)) - arma::sum((counts + rho_prop) * arma::log(1.0 + mean_cur * alpha_prop)) - n_sample * lgamma(rho_prop) + arma::sum(counts * log(alpha_prop));

  mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) && it_num > 50){
    // if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}


//    Function to update NB dispersion parameter for a single feature using
//    a random walk proposal and gamma prior

double update_rho_gam(const arma::rowvec &beta_cur,
                      const arma::rowvec &counts,
                      const double &alpha_cur,
                      const arma::vec &log_offset,
                      const arma::mat &design_mat,
                      const double &prior_shape,
                      const double &prior_scale,
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

  // mh_cur = ll_cur + R::dnorm4(log(alpha_cur), mean_alpha_cur, prior_sd_rs, 1);
  // mh_prop = ll_prop + R::dnorm4(log(alpha_prop), mean_alpha_cur, prior_sd_rs, 1);

  mh_cur = ll_cur + R::dgamma(rho_cur, prior_shape, prior_scale, 1);
  mh_prop = ll_prop + R::dgamma(rho_prop, prior_shape, prior_scale, 1);

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    return alpha_prop;
  }
  else{
    return alpha_cur;
  }
}

//   Function to run an entire chain for one feature
arma::field<arma::mat> whole_chain_nbglm(const arma::rowvec &counts,
                                         const arma::vec &log_offset,
                                         const arma::rowvec &starting_betas,
                                         const arma::mat &design_mat,
                                         const double &mean_rho,
                                         const double &prior_sd_betas,
                                         const double &prior_sd_rs,
                                         const double &rw_sd_rs,
                                         const double &n_beta,
                                         const double &n_sample,
                                         const int n_it){
  int i, accepts = 0;
  arma::field<arma::mat> ret(1, 3);
  arma::mat betas_sample(n_it, n_beta);
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

  for(i = 1; i < n_it; i++){
    betas_cur = arma::trans(update_betas_wls(betas_last,
                                             counts,
                                             disp_sample(i-1),
                                             log_offset,
                                             design_mat,
                                             prior_sd_betas,
                                             n_beta,
                                             n_sample,
                                             R_mat_diag,
                                             accepts));
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
  }
  ret(0, 0) = betas_sample;
  ret(0, 1) = disp_sample;
  ret(0, 2) = accepts;
  return(ret);
}

struct whole_feature_sample_struct_glm : public Worker
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
  const int &n_beta;
  const int &n_sample;
  const int &n_it;

  // density that I have accumulated
  arma::cube &upd_betas;
  arma::cube &upd_alphas;
  arma::field<arma::mat> mcmc_res_tmp;
  arma::cube &upd_accepts;

  // constructors
  whole_feature_sample_struct_glm(const arma::mat &counts,
                                  const arma::vec &log_offset,
                                  const arma::mat &starting_betas,
                                  const arma::mat &design_mat,
                                  const arma::vec &mean_rhos,
                                  const double &prior_sd_rs,
                                  const double &rw_sd_rs,
                                  const double &prior_sd_betas,
                                  const int &n_beta,
                                  const int &n_sample,
                                  const int &n_it,
                                  arma::cube &upd_betas,
                                  arma::cube &upd_alphas,
                                  arma::cube &upd_accepts)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      n_beta(n_beta), n_sample(n_sample), n_it(n_it), upd_betas(upd_betas), upd_alphas(upd_alphas),
      upd_accepts(upd_accepts){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      mcmc_res_tmp = whole_chain_nbglm(counts.row(i),
                                       log_offset,
                                       starting_betas.row(i),
                                       design_mat,
                                       mean_rhos(i),
                                       prior_sd_betas,
                                       prior_sd_rs,
                                       rw_sd_rs,
                                       n_beta,
                                       n_sample,
                                       n_it);
      // Rcpp::Rcout << "Line 3124 check" << std::endl;
      upd_betas.slice(i) = mcmc_res_tmp(0, 0);
      // Rcpp::Rcout << "Line 3126 check" << std::endl;
      upd_alphas.slice(0).col(i) = mcmc_res_tmp(0, 1);
      // Rcpp::Rcout << "Line 3128 check" << std::endl;
      // Rcpp::Rcout << "Line 3130 check" << std::endl;
      upd_accepts(i, 0, 0) = mcmc_res_tmp(0, 2)(0, 0);
      // Rcpp::Rcout << "Line 3132 check" << std::endl;
    }
  }

};

arma::field<arma::cube> mcmc_chain_glm_par(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_sample,
                                           const int &n_it){
  arma::cube upd_betas(n_it, n_beta, counts.n_rows);
  arma::cube upd_alphas(n_it, counts.n_rows, 1);
  arma::cube upd_accepts(counts.n_rows, 1, 1);
  arma::field<arma::cube> ret(1, 3);

  whole_feature_sample_struct_glm mcmc_inst(counts,
                                            log_offset,
                                            starting_betas,
                                            design_mat,
                                            mean_rhos,
                                            prior_sd_rs,
                                            rw_sd_rs,
                                            prior_sd_betas,
                                            n_beta,
                                            n_sample,
                                            n_it,
                                            upd_betas,
                                            upd_alphas,
                                            upd_accepts);
  parallelFor(0, counts.n_rows, mcmc_inst);
  ret(0, 0) = upd_betas;
  ret(0, 1) = upd_alphas;
  ret(0, 2) = upd_accepts;
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(ret);
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

Rcpp::List nbglm_mcmc_fp(arma::mat counts,
                         arma::mat design_mat,
                         double prior_sd_betas,
                         double prior_sd_rs,
                         arma::vec prior_mean_log_rs,
                         int n_it,
                         double rw_sd_rs,
                         arma::vec log_offset,
                         arma::mat starting_betas,
                         int grain_size = 1){

  arma::field<arma::cube> ret;
  int n_beta = design_mat.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  arma::mat starting_betas2(counts.n_rows, n_beta);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;

  // arma::field<arma::cube> mcmc_chain_glm_par(const arma::mat &counts,
  //                                            const arma::vec &log_offset,
  //                                            const arma::mat &starting_betas,
  //                                            const arma::mat &design_mat,
  //                                            const arma::vec &mean_rhos,
  //                                            const double &prior_sd_rs,
  //                                            const double &rw_sd_rs,
  //                                            const double &prior_sd_betas,
  //                                            const int &n_beta,
  //                                            const int &n_sample,
  //                                            const int &n_it){
  ret = mcmc_chain_glm_par(counts,
                           log_offset,
                           starting_betas2,
                           design_mat,
                           prior_mean_log_rs,
                           prior_sd_rs,
                           rw_sd_rs,
                           prior_sd_betas,
                           n_beta,
                           n_sample,
                           n_it);
  // arma::vec accept_vec = ret(0, 3).slice(0).col(0);
  // arma::uvec idx_na = arma::find(accept_vec < 0);
  // int n_na = idx_na.n_elem;
  // if(n_na > 0){
  //   arma::vec na_tmp(n_na);
  //   na_tmp.fill(NA_REAL);
  //   accept_vec.rows(idx_na) = na_tmp;
  // }

  return Rcpp::List::create(Rcpp::Named("betas_sample") = ret(0, 0),
                            Rcpp::Named("alphas_sample") = ret(0, 1).slice(0),
                            Rcpp::Named("accepts") = ret(0, 2).slice(0));
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

//  Structure for updating of regression parameters using WLS proposal

struct upd_betas_wls_struct : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const int &n_beta;
  const int &n_sample;
  const arma::vec &R_mat_diag;
  arma::ivec &accept_rec_vec;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_wls_struct(const arma::mat &beta_cur,
                       const arma::mat &counts,
                       const arma::vec &alpha_cur,
                       const arma::vec &log_offset,
                       const arma::mat &design_mat,
                       const double &prior_sd_betas,
                       const int &n_beta,
                       const int &n_sample,
                       const arma::vec &R_mat_diag,
                       arma::ivec &accept_rec_vec,
                       arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), n_beta(n_beta),
      n_sample(n_sample), R_mat_diag(R_mat_diag), accept_rec_vec(accept_rec_vec), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_wls(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset,
                                     design_mat, prior_sd_betas, n_beta, n_sample, R_mat_diag, accept_rec_vec(i)));
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
                        arma::vec &upd_rhos)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), mean_alpha_cur(mean_alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), n_beta(n_beta), n_sample(n_sample), it_num(it_num), upd_rhos(upd_rhos) {}

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
               it_num);
    }
  }

};

//  Structure for parallel updating of NB dispersions using random walk and gamma prior

struct upd_rhos_gam_struct : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const arma::vec &prior_shape;
  const arma::vec &prior_scale;
  const double &rw_sd_rs;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::vec &upd_rhos;

  // constructors
  upd_rhos_gam_struct(const arma::mat &beta_cur,
                      const arma::mat &counts,
                      const arma::vec &alpha_cur,
                      const arma::vec &log_offset,
                      const arma::mat &design_mat,
                      const arma::vec &prior_shape,
                      const arma::vec &prior_scale,
                      const double &rw_sd_rs,
                      const int &n_beta,
                      const int &n_sample,
                      arma::vec &upd_rhos)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_shape(prior_shape), prior_scale(prior_scale), rw_sd_rs(rw_sd_rs), n_beta(n_beta), n_sample(n_sample),
      upd_rhos(upd_rhos) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_rhos(i) = update_rho_gam(beta_cur.row(i),
               counts.row(i),
               alpha_cur(i),
               log_offset,
               design_mat,
               prior_shape(i),
               prior_scale(i),
               rw_sd_rs,
               n_beta,
               n_sample);
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

//  Parallel updating of regression coeficients using WLS proposal
arma::mat para_update_betas_wls(const arma::mat &beta_cur,
                                const arma::mat &counts,
                                const arma::vec &alpha_cur,
                                const arma::vec &log_offset,
                                const arma::mat &design_mat,
                                const double &prior_sd_betas,
                                const int &n_beta,
                                const int &n_sample,
                                const arma::vec &R_mat_diag,
                                arma::ivec &accept_rec_vec,
                                const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_wls_struct upd_betas_inst(beta_cur,
                                      counts,
                                      alpha_cur,
                                      log_offset,
                                      design_mat,
                                      prior_sd_betas,
                                      n_beta,
                                      n_sample,
                                      R_mat_diag,
                                      accept_rec_vec,
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
                                      upd_rhos);
  parallelFor(0, counts.n_rows, upd_rhos_inst, grain_size);
  return(upd_rhos);
}

//  Parallel updating of NB dispersions using random walk proposal with gamma prior
arma::vec para_update_rhos_gam(const arma::mat &beta_cur,
                               const arma::mat &counts,
                               const arma::vec &alpha_cur,
                               const arma::vec &log_offset,
                               const arma::mat &design_mat,
                               const arma::vec &prior_shape,
                               const arma::vec &prior_scale,
                               const double &rw_sd_rs,
                               const int &n_beta,
                               const int &n_sample,
                               const int &grain_size){
  arma::vec upd_rhos(counts.n_rows);

  upd_rhos_gam_struct upd_rhos_inst(beta_cur,
                                    counts,
                                    alpha_cur,
                                    log_offset,
                                    design_mat,
                                    prior_shape,
                                    prior_scale,
                                    rw_sd_rs,
                                    n_beta,
                                    n_sample,
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


//' Negative Binomial GLM MCMC WLS (title)
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
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for regression coefficients (n_feature x n_beta)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters and a matrix of dispersion values
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc_wls(arma::mat counts,
                          arma::mat design_mat,
                          double prior_sd_betas,
                          double prior_sd_rs,
                          arma::vec prior_mean_log_rs,
                          int n_it,
                          double rw_sd_rs,
                          arma::vec log_offset,
                          arma::mat starting_betas,
                          int grain_size = 1){
  int i, n_beta = design_mat.n_cols, n_feature = counts.n_rows, n_sample = counts.n_cols;
  arma::cube betas(n_feature, n_beta, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta);
  arma::vec prior_mean_betas(n_beta);
  arma::ivec accept_rec_vec(n_feature);
  arma::vec beta_cur(n_feature), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);

  betas.zeros();
  betas.slice(0) = starting_betas;
  rhos.row(0) = arma::trans(arma::exp(prior_mean_log_rs));
  prior_mean_betas.zeros();
  arma::vec R_mat_diag(n_beta);
  accept_rec_vec.zeros();

  mean_rho_cur = prior_mean_log_rs;
  R_mat_diag.zeros();
  R_mat_diag += pow(prior_sd_betas, 2);

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_wls(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat, prior_sd_betas, n_beta, n_sample, R_mat_diag, accept_rec_vec, grain_size);
    rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat, prior_sd_rs, rw_sd_rs, n_beta, n_sample, grain_size));

  }

  // Return list with posterior samples
  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                            Rcpp::Named("alphas_sample") = rhos,
                            Rcpp::Named("accepts") = accept_rec_vec);
}


//' Negative Binomial GLM MCMC WLS Gamma-disp (title)
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
//' @param prior_shape vector of prior gamma shape parameters for dispersions
//' @param prior_scale vector of prior gamma scale parameters for dispersions
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random walk std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for regression coefficients (n_feature x n_beta)
//' @param starting_disps vector of starting values for dispersion parameters
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters and a matrix of dispersion values
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbglm_mcmc_wls_gam(arma::mat counts,
                              arma::mat design_mat,
                              double prior_sd_betas,
                              arma::vec prior_shape,
                              arma::vec prior_scale,
                              int n_it,
                              double rw_sd_rs,
                              arma::vec log_offset,
                              arma::mat starting_betas,
                              arma::vec starting_disps,
                              int grain_size = 1){
  int i, n_beta = design_mat.n_cols, n_feature = counts.n_rows, n_sample = counts.n_cols;
  arma::cube betas(n_feature, n_beta, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta);
  arma::vec prior_mean_betas(n_beta);
  arma::ivec accept_rec_vec(n_feature);
  arma::vec beta_cur(n_feature), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);

  betas.zeros();
  betas.slice(0) = starting_betas;
  rhos.row(0) = arma::trans(starting_disps);
  prior_mean_betas.zeros();
  arma::vec R_mat_diag(n_beta);
  accept_rec_vec.zeros();

  R_mat_diag.zeros();
  R_mat_diag += pow(prior_sd_betas, 2);

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_wls(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat, prior_sd_betas, n_beta, n_sample, R_mat_diag, accept_rec_vec, grain_size);
    rhos.row(i) = arma::trans(para_update_rhos_gam(betas.slice(i), counts, rhos.row(i-1).t(), log_offset, design_mat, prior_shape, prior_scale, rw_sd_rs, n_beta, n_sample, grain_size));

  }

  // Return list with posterior samples
  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                            Rcpp::Named("alphas_sample") = rhos,
                            Rcpp::Named("accepts") = accept_rec_vec);
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
//  using a WLS proposal (slightly different than NBGLM version)

arma::vec update_betas_wls_mm(const arma::rowvec &beta_cur,
                              const arma::rowvec &counts,
                              const double &alpha_cur,
                              const arma::vec &log_offset,
                              const arma::mat &design_mat,
                              const double &prior_sd_betas,
                              const double &re_var,
                              const int &n_beta,
                              const int &n_beta_re,
                              const int &n_sample,
                              int &accept_rec){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta + n_beta_re), mean_wls_prop(n_beta + n_beta_re);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re), cov_mat_prop(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re);

  R_mat_diag.zeros();
  // R_mat_diag.fill(prior_var_betas);
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;
  prior_mean_betas.zeros();
  //Rcpp::Rcout << "alpha_cur = " << alpha_cur << std::endl;
  //Rcpp::Rcout << "beta_cur = " << beta_cur_tmp << std::endl;
  eta = design_mat * beta_cur_tmp + log_offset;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;

  cov_mat_cur = arma::inv(R_mat.i() + design_mat.t() * W_mat * design_mat);
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat.t() * W_mat * y_tilde);
  // Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;
  // Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur.size() << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  // Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  // Rcpp::Rcout << "beta_prop = " << beta_prop.size() << std::endl;
  eta_prop = design_mat * beta_prop + log_offset;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);

  W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  cov_mat_prop = arma::inv(R_mat.i() + design_mat.t() * W_mat_prop * design_mat);
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat.t() * W_mat_prop * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  // Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  // Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  // Rcpp::Rcout << "log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) = " << log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) << std::endl;
  // Rcpp::Rcout << "-log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop)) = " << log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop)) << std::endl;
  // Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  // Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  // Rcpp::Rcout << "log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) = " << log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) << std::endl;
  // Rcpp::Rcout << "-log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur)) = " << log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur)) << std::endl;
  // Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  // Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;
  // Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur.diag().max() << std::endl;
  // Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}

//  Function for simultaneously updating fixed effects and random intercepts
//  using a WLS proposal (slightly different than NBGLM version) with forced
//  accepts for the first 50 iterations

arma::vec update_betas_wls_mm_force(const arma::rowvec &beta_cur,
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
                                    int &inv_errors){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta + n_beta_re), mean_wls_prop(n_beta + n_beta_re);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re), cov_mat_prop(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re);
  bool success = false;

  R_mat_diag.zeros();
  // R_mat_diag.fill(prior_var_betas);
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;
  prior_mean_betas.zeros();
  //Rcpp::Rcout << "alpha_cur = " << alpha_cur << std::endl;
  // Rcpp::Rcout << "beta_cur = " << beta_cur_tmp << std::endl;
  // Rcpp::Rcout << "line 1245 check" << std::endl;
  eta = design_mat * beta_cur_tmp + log_offset;
  // Rcpp::Rcout << "line 1247 check" << std::endl;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  arma::mat W_mat_i;
  success = false;
  while(success == false)
  {
    success = arma::inv(W_mat_i, W_mat);

    if(success == false)
    {
      inv_errors++;
      return(beta_cur_tmp);
    }
  }
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  //W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat inv ok " << std::endl;
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  // Rcpp::Rcout << "line 1274 check" << std::endl;
  arma::mat csigma = R_mat.i() + design_mat.t() * W_mat_i * design_mat;

  success = false;
  while(success == false)
  {
    success = arma::inv(cov_mat_cur, csigma);

    if(success == false)
    {
      inv_errors++;
      return(beta_cur_tmp);
    }
  }

  //cov_mat_cur = arma::inv(R_mat.i() + design_mat.t() * W_mat * design_mat);
  //Rcpp::Rcout << "cov_mat_cur inv ok " << std::endl;
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  // Rcpp::Rcout << "line 1292 check" << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat.t() * W_mat_i * y_tilde);
  //Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;
  // Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur.size() << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  //Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  // Rcpp::Rcout << "beta_prop = " << beta_prop.size() << std::endl;
  // Rcpp::Rcout << "line 1300 check" << std::endl;
  eta_prop = design_mat * beta_prop + log_offset;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) - log_offset;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);
  //Rcpp::Rcout << "alpha_cur = " << alpha_cur << std::endl;
  //Rcpp::Rcout << "W_mat_prop_diag = " << W_mat_prop.diag() << std::endl;
  arma::mat W_mat_prop_i;
  success = false;
  while(success == false)
  {
    success = arma::inv(W_mat_prop_i, W_mat_prop);

    if(success == false)
    {
      inv_errors++;
      return(beta_cur_tmp);
    }
  }

  //W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop inv ok " << std::endl;
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  // Rcpp::Rcout << "line 1331 check" << std::endl;
  csigma = R_mat.i() + design_mat.t() * W_mat_prop_i * design_mat;
  success = false;

  while(success == false)
  {
    success = arma::inv(cov_mat_prop, csigma);

    if(success == false)
    {
      inv_errors++;
      return(beta_cur_tmp);
    }
  }
  //cov_mat_prop = arma::inv(R_mat.i() + design_mat.t() * W_mat_prop * design_mat);
  //Rcpp::Rcout << "cov_mat_prop inv ok " << std::endl;
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  // Rcpp::Rcout << "line 1348 check" << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat.t() * W_mat_prop_i * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  // Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  // Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  // Rcpp::Rcout << "log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) = " << log(dmvnrm_1f(beta_cur_tmp, prior_mean_betas, R_mat)) << std::endl;
  // Rcpp::Rcout << "-log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop)) = " << log(dmvnrm_1f(beta_cur_tmp, mean_wls_prop, cov_mat_prop)) << std::endl;
  // Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  // Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  // Rcpp::Rcout << "log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) = " << log(dmvnrm_1f(beta_prop, prior_mean_betas, R_mat)) << std::endl;
  // Rcpp::Rcout << "-log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur)) = " << log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur)) << std::endl;
  // Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  // Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;
  // Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur.diag().max() << std::endl;
  // Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if((R::runif(0, 1) < exp(mh_prop - mh_cur)) || it_num <= 50){
    beta_cur_tmp = beta_prop;
    accept_rec += 1;
  }
  return beta_cur_tmp;
}


//  Function for sequentially updating fixed effects and random intercepts
//  using a WLS proposal (slightly different than NBGLM version)

arma::vec update_betas_wls_mm_split(const arma::rowvec &beta_cur,
                                    const arma::rowvec &counts,
                                    const double &alpha_cur,
                                    const arma::vec &log_offset,
                                    const arma::mat &design_mat,
                                    const double &prior_sd_betas,
                                    const double &re_var,
                                    const int &n_beta,
                                    const int &n_beta_re,
                                    const int &n_sample){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta), mean_wls_prop(n_beta);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re), cov_mat_prop(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re), beta_cur_fe, beta_cur_re;
  arma::mat design_mat_fe, design_mat_re, R_mat_fe, R_mat_re;

  design_mat_fe = design_mat.cols(0, n_beta - 1);
  design_mat_re = design_mat.cols(n_beta, n_beta + n_beta_re - 1);

  beta_cur_fe = beta_cur_tmp.rows(0, n_beta - 1);
  beta_cur_re = beta_cur_tmp.rows(n_beta, n_beta + n_beta_re - 1);

  R_mat_diag.zeros();
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;

  prior_mean_betas.zeros();
  //Rcpp::Rcout << "alpha_cur = " << alpha_cur << std::endl;
  //Rcpp::Rcout << "beta_cur = " << beta_cur_tmp << std::endl;
  eta = design_mat * beta_cur_tmp + log_offset;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset - design_mat_re * beta_cur_re;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  R_mat_fe = R_mat.rows(0, n_beta - 1).cols(0, n_beta - 1);
  R_mat_re = R_mat.rows(n_beta, n_beta + n_beta_re - 1).cols(n_beta, n_beta + n_beta_re - 1);

  cov_mat_cur = arma::inv(R_mat_fe.i() + design_mat_fe.t() * W_mat * design_mat_fe);
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat_fe.t() * W_mat * y_tilde);
  //Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  //Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  eta_prop = design_mat_fe * beta_prop + log_offset + design_mat_re * beta_cur_re;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) -
    log_offset - design_mat_re * beta_cur_re;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);

  W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  cov_mat_prop = arma::inv(R_mat_fe.i() + design_mat_fe.t() * W_mat_prop * design_mat_fe);
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat_fe.t() * W_mat_prop * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_fe, prior_mean_betas.rows(0, n_beta - 1), R_mat_fe)) -
    log(dmvnrm_1f(beta_cur_fe, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas.rows(0, n_beta - 1), R_mat_fe)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  //Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  //Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  //Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  //Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp.rows(0, n_beta - 1) = beta_prop;
    beta_cur_fe = beta_prop;
  }

  /*
   * Now doing the Random effects as a seperate block
   */

  eta = design_mat * beta_cur_tmp + log_offset;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset - design_mat_fe * beta_cur_fe;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;

  cov_mat_cur = arma::inv(R_mat_re.i() + design_mat_re.t() * W_mat * design_mat_re);
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat_re.t() * W_mat * y_tilde);
  //Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  //Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  eta_prop = design_mat_re * beta_prop + log_offset + design_mat_fe * beta_cur_fe;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) -
    log_offset - design_mat_fe * beta_cur_fe;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);

  W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  cov_mat_prop = arma::inv(R_mat_re.i() + design_mat_re.t() * W_mat_prop * design_mat_re);
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat_re.t() * W_mat_prop * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_re, prior_mean_betas.rows(n_beta, n_beta + n_beta_re - 1), R_mat_re)) -
    log(dmvnrm_1f(beta_cur_re, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas.rows(n_beta, n_beta + n_beta_re - 1), R_mat_re)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  //Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  //Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  //Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  //Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp.rows(n_beta, n_beta + n_beta_re - 1) = beta_prop;
  }

  //Return sample of betas
  return beta_cur_tmp;
}



//  Function for sequentially updating fixed effects and random intercepts
//  using a WLS proposal (slightly different than NBGLM version)

arma::vec update_betas_wls_mm_split_half(const arma::rowvec &beta_cur,
                                         const arma::rowvec &counts,
                                         const double &alpha_cur,
                                         const arma::vec &log_offset,
                                         const arma::mat &design_mat,
                                         const double &prior_sd_betas,
                                         const double &re_var,
                                         const int &n_beta,
                                         const int &n_beta_re,
                                         const int &n_sample){
  arma::vec beta_prop(n_beta + n_beta_re), mean_cur(n_sample), mean_prop(n_sample), beta_cur_tmp = beta_cur.t();
  arma::vec y_tilde(n_sample), y_tilde_prop(n_sample), eta(n_sample), eta_prop(n_sample);
  arma::vec mean_wls_cur(n_beta), mean_wls_prop(n_beta);
  arma::mat W_mat(n_sample, n_sample), R_mat(n_beta + n_beta_re, n_beta + n_beta_re);
  arma::mat W_mat_prop(n_sample, n_sample);
  arma::mat cov_mat_cur(n_beta + n_beta_re, n_beta + n_beta_re), cov_mat_prop(n_beta + n_beta_re, n_beta + n_beta_re);
  double ll_prop, ll_cur, mh_prop, mh_cur, rho_cur = 1.0 / alpha_cur;
  double prior_var_betas = pow(prior_sd_betas, 2);
  arma::vec prior_mean_betas(n_beta + n_beta_re), R_mat_diag(n_beta + n_beta_re), beta_cur_fe, beta_cur_re;
  arma::mat design_mat_fe, design_mat_re, R_mat_fe, R_mat_re;
  int n_beta_split = (n_beta + n_beta_re) / 2;

  design_mat_fe = design_mat.cols(0, n_beta_split - 1);
  design_mat_re = design_mat.cols(n_beta_split, n_beta + n_beta_re - 1);

  beta_cur_fe = beta_cur_tmp.rows(0, n_beta_split - 1);
  beta_cur_re = beta_cur_tmp.rows(n_beta_split, n_beta + n_beta_re - 1);

  R_mat_diag.zeros();
  R_mat_diag.rows(0, n_beta - 1) += prior_var_betas;
  R_mat_diag.rows(n_beta, n_beta + n_beta_re -1) += re_var;

  prior_mean_betas.zeros();
  //Rcpp::Rcout << "alpha_cur = " << alpha_cur << std::endl;
  //Rcpp::Rcout << "beta_cur = " << beta_cur_tmp << std::endl;
  eta = design_mat * beta_cur_tmp + log_offset;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset - design_mat_re * beta_cur_re;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;
  R_mat.zeros();
  R_mat.diag() = R_mat_diag;
  R_mat_fe = R_mat.rows(0, n_beta_split - 1).cols(0, n_beta_split - 1);
  R_mat_re = R_mat.rows(n_beta_split, n_beta + n_beta_re - 1).cols(n_beta_split, n_beta + n_beta_re - 1);

  cov_mat_cur = arma::inv(R_mat_fe.i() + design_mat_fe.t() * W_mat * design_mat_fe);
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat_fe.t() * W_mat * y_tilde);
  //Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  //Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  eta_prop = design_mat_fe * beta_prop + log_offset + design_mat_re * beta_cur_re;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) -
    log_offset - design_mat_re * beta_cur_re;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);

  W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  cov_mat_prop = arma::inv(R_mat_fe.i() + design_mat_fe.t() * W_mat_prop * design_mat_fe);
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat_fe.t() * W_mat_prop * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_fe, prior_mean_betas.rows(0, n_beta_split - 1), R_mat_fe)) -
    log(dmvnrm_1f(beta_cur_fe, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas.rows(0, n_beta_split - 1), R_mat_fe)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  //Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  //Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  //Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  //Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp.rows(0, n_beta_split - 1) = beta_prop;
    beta_cur_fe = beta_prop;
  }

  /*
   * Now doing the Random effects as a seperate block
   */

  eta = design_mat * beta_cur_tmp + log_offset;
  //eta = design_mat * beta_cur_tmp;
  //Rcpp::Rcout << "eta = " << eta << std::endl;
  mean_cur = arma::exp(eta);
  //Rcpp::Rcout << "mean_cur = " << mean_cur << std::endl;
  //y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta);
  y_tilde = eta + arma::trans(counts - mean_cur.t()) % arma::exp(-eta) - log_offset - design_mat_fe * beta_cur_fe;
  //Rcpp::Rcout << "y_tilde = " << y_tilde << std::endl;
  W_mat.zeros();
  W_mat.diag() = alpha_cur + arma::exp(-eta);
  //W_mat = arma::diagmat(alpha_cur + arma::exp(-eta));
  W_mat = W_mat.i();
  //Rcpp::Rcout << "W_mat = " << W_mat << std::endl;

  cov_mat_cur = arma::inv(R_mat_re.i() + design_mat_re.t() * W_mat * design_mat_re);
  //Rcpp::Rcout << "cov_mat_cur = " << cov_mat_cur << std::endl;
  mean_wls_cur = cov_mat_cur * (design_mat_re.t() * W_mat * y_tilde);
  //Rcpp::Rcout << "mean_wls_cur = " << mean_wls_cur << std::endl;

  beta_prop = arma::trans(rmvnormal(1, mean_wls_cur, cov_mat_cur));
  //Rcpp::Rcout << "beta_prop = " << beta_prop << std::endl;
  eta_prop = design_mat_re * beta_prop + log_offset + design_mat_fe * beta_cur_fe;
  //eta_prop = design_mat * beta_prop;
  //Rcpp::Rcout << "eta_prop = " << eta_prop << std::endl;
  mean_prop = arma::exp(eta_prop);
  //Rcpp::Rcout << "mean_prop = " << mean_prop << std::endl;

  //y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop);
  y_tilde_prop = eta_prop + arma::trans(counts - mean_prop.t()) % arma::exp(-eta_prop) -
    log_offset - design_mat_fe * beta_cur_fe;
  //Rcpp::Rcout << "y_tilde_prop = " << y_tilde_prop << std::endl;

  W_mat_prop.zeros();
  W_mat_prop.diag() = alpha_cur + arma::exp(-eta_prop);

  W_mat_prop = W_mat_prop.i();
  //Rcpp::Rcout << "W_mat_prop = " << W_mat_prop << std::endl;
  cov_mat_prop = arma::inv(R_mat_re.i() + design_mat_re.t() * W_mat_prop * design_mat_re);
  //Rcpp::Rcout << "cov_mat_prop = " << cov_mat_prop << std::endl;
  mean_wls_prop = cov_mat_prop * (design_mat_re.t() * W_mat_prop * y_tilde_prop);
  //Rcpp::Rcout << "mean_wls_prop = " << mean_wls_prop << std::endl;

  ll_cur = arma::sum(counts * arma::log(mean_cur) - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur));
  //Rcpp::Rcout << "counts * arma::log(mean_cur)" << counts * arma::log(mean_cur) << std::endl;
  //Rcpp::Rcout << "- (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur)" << - (counts + rho_cur) * arma::log(1.0 + mean_cur * alpha_cur) << std::endl;
  ll_prop = arma::sum(counts * arma::log(mean_prop) - (counts + rho_cur) * arma::log(1.0 + mean_prop * alpha_cur));
  //Rcpp::Rcout << "ll_cur = " << ll_cur << std::endl;
  //Rcpp::Rcout << "ll_prop = " << ll_prop << std::endl;
  mh_cur = ll_cur +
    log(dmvnrm_1f(beta_cur_re, prior_mean_betas.rows(n_beta_split, n_beta + n_beta_re - 1), R_mat_re)) -
    log(dmvnrm_1f(beta_cur_re, mean_wls_prop, cov_mat_prop));

  mh_prop = ll_prop +
    log(dmvnrm_1f(beta_prop, prior_mean_betas.rows(n_beta_split, n_beta + n_beta_re - 1), R_mat_re)) -
    log(dmvnrm_1f(beta_prop, mean_wls_cur, cov_mat_cur));

  //Rcpp::Rcout << "mh_cur = " << mh_cur << std::endl;
  //Rcpp::Rcout << "mh_prop = " << mh_prop << std::endl;
  //Rcpp::Rcout << "mh_diff = " << mh_prop - mh_cur << std::endl;
  //Rcpp::Rcout << "mh_accept_prob = " << exp(mh_prop - mh_cur) << std::endl;

  if(R::runif(0, 1) < exp(mh_prop - mh_cur)){
    beta_cur_tmp.rows(n_beta_split, n_beta + n_beta_re - 1) = beta_prop;
  }

  //Return sample of betas
  return beta_cur_tmp;
}


////////////////////////////////////////////////////////////////////////////////
/*
 * Declaration of structures used in RcppParallel to update features in parallel
 */
////////////////////////////////////////////////////////////////////////////////

//  Structure to update fixed effects using random walk proposal
struct upd_betas_struct_mm : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::mat &beta_re;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const arma::mat &design_mat_re;
  const double &prior_sd_betas;
  const double &rw_sd_betas;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_struct_mm(const arma::mat &beta_cur,
                      const arma::mat &counts,
                      const arma::mat &beta_re,
                      const arma::vec &alpha_cur,
                      const arma::vec &log_offset,
                      const arma::mat &design_mat,
                      const arma::mat &design_mat_re,
                      const double &prior_sd_betas,
                      const double &rw_sd_betas,
                      const int &n_beta,
                      const int &n_sample,
                      arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), beta_re(beta_re), alpha_cur(alpha_cur), log_offset(log_offset),
      design_mat(design_mat), design_mat_re(design_mat_re), prior_sd_betas(prior_sd_betas),
      rw_sd_betas(rw_sd_betas), n_beta(n_beta), n_sample(n_sample), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_fe(beta_cur.row(i), counts.row(i), beta_re.row(i), alpha_cur(i),
                                     log_offset, design_mat, design_mat_re, prior_sd_betas, rw_sd_betas, n_beta,
                                     n_sample));
    }
  }

};

//  Structure to update random effects using random walk proposal
struct upd_betas_re_struct : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::mat &beta_fe;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const arma::mat &design_mat_re;
  const arma::rowvec &prior_sd_betas;
  const double &rw_sd_betas;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_re_struct(const arma::mat &beta_cur,
                      const arma::mat &counts,
                      const arma::mat &beta_fe,
                      const arma::vec &alpha_cur,
                      const arma::vec &log_offset,
                      const arma::mat &design_mat,
                      const arma::mat &design_mat_re,
                      const arma::rowvec &prior_sd_betas,
                      const double &rw_sd_betas,
                      const int &n_beta,
                      const int &n_sample,
                      arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), beta_fe(beta_fe), alpha_cur(alpha_cur), log_offset(log_offset),
      design_mat(design_mat), design_mat_re(design_mat_re), prior_sd_betas(prior_sd_betas),
      rw_sd_betas(rw_sd_betas), n_beta(n_beta), n_sample(n_sample), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_re(beta_cur.row(i), counts.row(i), beta_fe.row(i), alpha_cur(i),
                                     log_offset, design_mat, design_mat_re, sqrt(prior_sd_betas(i)), rw_sd_betas, n_beta,
                                     n_sample));
    }
  }

};

//  Structure to update NB dispersion (for use with RW updates for regression parameters)
struct upd_rhos_struct_mm : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::mat &beta_re;
  const arma::vec &alpha_cur;
  const arma::vec &mean_alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const arma::mat &design_mat_re;
  const double &prior_sd_rs;
  const double &rw_sd_rs;
  const int &n_beta;
  const int &n_sample;

  // updates accumulated so far
  arma::vec &upd_rhos;

  // constructors
  upd_rhos_struct_mm(const arma::mat &beta_cur,
                     const arma::mat &counts,
                     const arma::mat &beta_re,
                     const arma::vec &alpha_cur,
                     const arma::vec &mean_alpha_cur,
                     const arma::vec &log_offset,
                     const arma::mat &design_mat,
                     const arma::mat &design_mat_re,
                     const double &prior_sd_rs,
                     const double &rw_sd_rs,
                     const int &n_beta,
                     const int &n_sample,
                     arma::vec &upd_rhos)
    : beta_cur(beta_cur), counts(counts), beta_re(beta_re), alpha_cur(alpha_cur), mean_alpha_cur(mean_alpha_cur),
      log_offset(log_offset), design_mat(design_mat), design_mat_re(design_mat_re), prior_sd_rs(prior_sd_rs),
      rw_sd_rs(rw_sd_rs), n_beta(n_beta), n_sample(n_sample), upd_rhos(upd_rhos) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_rhos(i) = update_rho_mm(beta_cur.row(i),
               counts.row(i),
               beta_re.row(i),
               alpha_cur(i),
               mean_alpha_cur(i),
               log_offset,
               design_mat,
               design_mat_re,
               prior_sd_rs,
               rw_sd_rs,
               n_beta,
               n_sample);
    }
  }

};

//  Structure to update regression coefficients and random effects simultaneously
//  using the WLS proposal
struct upd_betas_wls_struct_mm : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const arma::rowvec &re_var;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  arma::ivec &accept_rec_vec;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_wls_struct_mm(const arma::mat &beta_cur,
                          const arma::mat &counts,
                          const arma::vec &alpha_cur,
                          const arma::vec &log_offset,
                          const arma::mat &design_mat,
                          const double &prior_sd_betas,
                          const arma::rowvec &re_var,
                          const int &n_beta,
                          const int &n_beta_re,
                          const int &n_sample,
                          arma::ivec &accept_rec_vec,
                          arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), re_var(re_var), n_beta(n_beta), n_beta_re(n_beta_re),
      n_sample(n_sample), accept_rec_vec(accept_rec_vec), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_wls_mm(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset,
                                     design_mat, prior_sd_betas, re_var(i), n_beta,n_beta_re, n_sample,
                                     accept_rec_vec(i)));
    }
  }

};

//  Structure to update regression coefficients and random effects simultaneously
//  using the WLS proposal with forced accepts for first 100 iterations
struct upd_betas_wls_struct_mm_force : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const arma::rowvec &re_var;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  arma::ivec &accept_rec_vec;
  const int &it_num;
  arma::ivec &inv_errors_vec;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_wls_struct_mm_force(const arma::mat &beta_cur,
                                const arma::mat &counts,
                                const arma::vec &alpha_cur,
                                const arma::vec &log_offset,
                                const arma::mat &design_mat,
                                const double &prior_sd_betas,
                                const arma::rowvec &re_var,
                                const int &n_beta,
                                const int &n_beta_re,
                                const int &n_sample,
                                arma::ivec &accept_rec_vec,
                                const int &it_num,
                                arma::ivec &inv_errors_vec,
                                arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), re_var(re_var), n_beta(n_beta), n_beta_re(n_beta_re),
      n_sample(n_sample), accept_rec_vec(accept_rec_vec), it_num(it_num), inv_errors_vec(inv_errors_vec),
      upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_wls_mm_force(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset,
                                     design_mat, prior_sd_betas, re_var(i), n_beta,n_beta_re, n_sample,
                                     accept_rec_vec(i), it_num, inv_errors_vec(i)));
    }
  }

};

//  Structure to update regression coefficients and random effects sequentially
//  using the WLS proposal
struct upd_betas_wls_struct_mm_split : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const arma::rowvec &re_var;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_wls_struct_mm_split(const arma::mat &beta_cur,
                                const arma::mat &counts,
                                const arma::vec &alpha_cur,
                                const arma::vec &log_offset,
                                const arma::mat &design_mat,
                                const double &prior_sd_betas,
                                const arma::rowvec &re_var,
                                const int &n_beta,
                                const int &n_beta_re,
                                const int &n_sample,
                                arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), re_var(re_var), n_beta(n_beta), n_beta_re(n_beta_re),
      n_sample(n_sample), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_wls_mm_split(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset,
                                     design_mat, prior_sd_betas, re_var(i), n_beta,n_beta_re,  n_sample));
    }
  }

};

//  Structure to update regression coefficients and random effects sequentially
//  using the WLS proposal (splitting total design mat into 2 pieces)
struct upd_betas_wls_struct_mm_split_half : public Worker
{
  // source objects
  const arma::mat &beta_cur;
  const arma::mat &counts;
  const arma::vec &alpha_cur;
  const arma::vec &log_offset;
  const arma::mat &design_mat;
  const double &prior_sd_betas;
  const arma::rowvec &re_var;
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;

  // updates accumulated so far
  arma::mat &upd_betas;

  // constructors
  upd_betas_wls_struct_mm_split_half(const arma::mat &beta_cur,
                                     const arma::mat &counts,
                                     const arma::vec &alpha_cur,
                                     const arma::vec &log_offset,
                                     const arma::mat &design_mat,
                                     const double &prior_sd_betas,
                                     const arma::rowvec &re_var,
                                     const int &n_beta,
                                     const int &n_beta_re,
                                     const int &n_sample,
                                     arma::mat &upd_betas)
    : beta_cur(beta_cur), counts(counts), alpha_cur(alpha_cur), log_offset(log_offset), design_mat(design_mat),
      prior_sd_betas(prior_sd_betas), re_var(re_var), n_beta(n_beta), n_beta_re(n_beta_re),
      n_sample(n_sample), upd_betas(upd_betas) {}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(unsigned int i = begin; i < end; i++){
      upd_betas.row(i) = arma::trans(update_betas_wls_mm_split_half(beta_cur.row(i), counts.row(i), alpha_cur(i), log_offset,
                                     design_mat, prior_sd_betas, re_var(i), n_beta,n_beta_re,  n_sample));
    }
  }

};
////////////////////////////////////////////////////////////////////////////////
/*
 * Parallelized functions for doing NBGLMM MCMC parameter updates using functions
 * and structures defined in previous 2 sections with parallelFor from RcppParallel
 */
////////////////////////////////////////////////////////////////////////////////

//  Parallel updating of fixed effect regression coeficients using random walk proposal
arma::mat para_update_betas_fe(const arma::mat &beta_cur,
                               const arma::mat &counts,
                               const arma::mat &beta_re,
                               const arma::vec &alpha_cur,
                               const arma::vec &log_offset,
                               const arma::mat &design_mat,
                               const arma::mat &design_mat_re,
                               const double &prior_sd_betas,
                               const double &rw_sd_betas,
                               const int &n_beta,
                               const int &n_sample,
                               const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_struct_mm upd_betas_inst(beta_cur,
                                     counts,
                                     beta_re,
                                     alpha_cur,
                                     log_offset,
                                     design_mat,
                                     design_mat_re,
                                     prior_sd_betas,
                                     rw_sd_betas,
                                     n_beta,
                                     n_sample,
                                     upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}

//  Parallel updating of random intercepts using random walk proposal
arma::mat para_update_betas_re(const arma::mat &beta_cur,
                               const arma::mat &counts,
                               const arma::mat &beta_fe,
                               const arma::vec &alpha_cur,
                               const arma::vec &log_offset,
                               const arma::mat &design_mat,
                               const arma::mat &design_mat_re,
                               const arma::rowvec &prior_sd_betas,
                               const double &rw_sd_betas,
                               const int &n_beta,
                               const int &n_sample,
                               const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_re_struct upd_betas_inst(beta_cur,
                                     counts,
                                     beta_fe,
                                     alpha_cur,
                                     log_offset,
                                     design_mat,
                                     design_mat_re,
                                     prior_sd_betas,
                                     rw_sd_betas,
                                     n_beta,
                                     n_sample,
                                     upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}

//  Parallel updating of NB dispersions using random walk proposal
//  (used only with RW updating version of NBGLMM)
arma::vec para_update_rhos_mm(const arma::mat &beta_cur,
                              const arma::mat &counts,
                              const arma::mat &beta_re,
                              const arma::vec &alpha_cur,
                              const arma::vec &mean_alpha_cur,
                              const arma::vec &log_offset,
                              const arma::mat &design_mat,
                              const arma::mat &design_mat_re,
                              const double &prior_sd_rs,
                              const double &rw_sd_rs,
                              const int &n_beta,
                              const int &n_sample,
                              const int &grain_size){
  arma::vec upd_rhos(counts.n_rows);

  upd_rhos_struct_mm upd_rhos_inst(beta_cur,
                                   counts,
                                   beta_re,
                                   alpha_cur,
                                   mean_alpha_cur,
                                   log_offset,
                                   design_mat,
                                   design_mat_re,
                                   prior_sd_rs,
                                   rw_sd_rs,
                                   n_beta,
                                   n_sample,
                                   upd_rhos);
  parallelFor(0, counts.n_rows, upd_rhos_inst, grain_size);
  return(upd_rhos);
}

//  Parallel updating of fixed and random effects using WLS proposal
arma::mat para_update_betas_wls_mm(const arma::mat &beta_cur,
                                   const arma::mat &counts,
                                   const arma::vec &alpha_cur,
                                   const arma::vec &log_offset,
                                   const arma::mat &design_mat,
                                   const double &prior_sd_betas,
                                   const arma::rowvec &re_var,
                                   const int &n_beta,
                                   const int &n_beta_re,
                                   const int &n_sample,
                                   arma::ivec &accept_rec_vec,
                                   const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_wls_struct_mm upd_betas_inst(beta_cur,
                                         counts,
                                         alpha_cur,
                                         log_offset,
                                         design_mat,
                                         prior_sd_betas,
                                         re_var,
                                         n_beta,
                                         n_beta_re,
                                         n_sample,
                                         accept_rec_vec,
                                         upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}

//  Parallel updating of fixed and random effects using WLS proposal
arma::mat para_update_betas_wls_mm_force(const arma::mat &beta_cur,
                                         const arma::mat &counts,
                                         const arma::vec &alpha_cur,
                                         const arma::vec &log_offset,
                                         const arma::mat &design_mat,
                                         const double &prior_sd_betas,
                                         const arma::rowvec &re_var,
                                         const int &n_beta,
                                         const int &n_beta_re,
                                         const int &n_sample,
                                         arma::ivec &accept_rec_vec,
                                         const int &it_num,
                                         arma::ivec &inv_errors_vec,
                                         const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_wls_struct_mm_force upd_betas_inst(beta_cur,
                                               counts,
                                               alpha_cur,
                                               log_offset,
                                               design_mat,
                                               prior_sd_betas,
                                               re_var,
                                               n_beta,
                                               n_beta_re,
                                               n_sample,
                                               accept_rec_vec,
                                               it_num,
                                               inv_errors_vec,
                                               upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}

//  Parallel updating of fixed and random effects using WLS proposal (sequential)
arma::mat para_update_betas_wls_mm_split(const arma::mat &beta_cur,
                                         const arma::mat &counts,
                                         const arma::vec &alpha_cur,
                                         const arma::vec &log_offset,
                                         const arma::mat &design_mat,
                                         const double &prior_sd_betas,
                                         const arma::rowvec &re_var,
                                         const int &n_beta,
                                         const int &n_beta_re,
                                         const int &n_sample,
                                         const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_wls_struct_mm_split upd_betas_inst(beta_cur,
                                               counts,
                                               alpha_cur,
                                               log_offset,
                                               design_mat,
                                               prior_sd_betas,
                                               re_var,
                                               n_beta,
                                               n_beta_re,
                                               n_sample,
                                               upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}

//  Parallel updating of fixed and random effects using WLS proposal (2 pieces)
arma::mat para_update_betas_wls_mm_split_half(const arma::mat &beta_cur,
                                              const arma::mat &counts,
                                              const arma::vec &alpha_cur,
                                              const arma::vec &log_offset,
                                              const arma::mat &design_mat,
                                              const double &prior_sd_betas,
                                              const arma::rowvec &re_var,
                                              const int &n_beta,
                                              const int &n_beta_re,
                                              const int &n_sample,
                                              const int &grain_size){
  arma::mat upd_betas(beta_cur.n_rows, beta_cur.n_cols);

  upd_betas_wls_struct_mm_split_half upd_betas_inst(beta_cur,
                                                    counts,
                                                    alpha_cur,
                                                    log_offset,
                                                    design_mat,
                                                    prior_sd_betas,
                                                    re_var,
                                                    n_beta,
                                                    n_beta_re,
                                                    n_sample,
                                                    upd_betas);
  parallelFor(0, counts.n_rows, upd_betas_inst, grain_size);
  return(upd_betas);
}
////////////////////////////////////////////////////////////////////////////////
/*
 * Wrapper functions to do MCMC for whole data set that are exported to R
 */
////////////////////////////////////////////////////////////////////////////////


//' Negative Binomial GLMM MCMC (title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_betas random walk std. dev. for proposing regression coefficients
//' @param rw_sd_betas_re random walk std. dev. for proposing random effects
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, a cube of random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler(arma::mat counts,
                             arma::mat design_mat,
                             arma::mat design_mat_re,
                             double prior_sd_betas,
                             double prior_sd_betas_a,
                             double prior_sd_betas_b,
                             double prior_sd_rs,
                             arma::vec prior_mean_log_rs,
                             int n_it,
                             double rw_sd_betas,
                             double rw_sd_betas_re,
                             double rw_sd_rs,
                             arma::vec log_offset,
                             int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols;
  arma::cube betas(n_feature, n_beta, n_it), betas_re(n_feature, n_beta_re, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  betas.zeros();
  rhos.row(0) = arma::exp(prior_mean_log_rs.t());

  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_fe(betas.slice(i-1), counts, betas_re.slice(i-1), rhos.row(i-1).t(), log_offset, design_mat, design_mat_re, prior_sd_betas, rw_sd_betas, n_beta, n_sample, grain_size);
    betas_re.slice(i) = para_update_betas_re(betas_re.slice(i-1), counts, betas.slice(i), rhos.row(i-1).t(), log_offset, design_mat, design_mat_re, sigma2.row(i-1), rw_sd_betas_re, n_beta_re, n_sample, grain_size);
    betas_cur_mat = betas_re.slice(i);
    rhos.row(i) = arma::trans(para_update_rhos_mm(betas.slice(i), counts, betas_re.slice(i), rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat, design_mat_re, prior_sd_rs, rw_sd_rs, n_beta, n_sample, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  // Return list with posterior samples
  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                            Rcpp::Named("rand_eff_sample") = betas_re,
                            Rcpp::Named("alphas_sample") = rhos,
                            Rcpp::Named("sigma2_sample") = sigma2);
}



//' Negative Binomial GLMM MCMC WLS(title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for fixed effects (n_feature x n_beta)
//' @param return_all_re logical variable to determine if posterior samples are returned for random effects (defaults to TRUE)
//' @param n_re_return number of random effects to return a full posterior sample for (defaults to 1, only used if return_all_re = FALSE)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls(arma::mat counts,
                                 arma::mat design_mat,
                                 arma::mat design_mat_re,
                                 double prior_sd_betas,
                                 double prior_sd_betas_a,
                                 double prior_sd_betas_b,
                                 double prior_sd_rs,
                                 arma::vec prior_mean_log_rs,
                                 int n_it,
                                 double rw_sd_rs,
                                 arma::vec log_offset,
                                 arma::mat starting_betas,
                                 bool return_all_re = true,
                                 int n_re_return = 1,
                                 int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols, n_beta_tot = n_beta + n_beta_re;
  arma::cube betas(n_feature, n_beta_tot, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  double n_beta_start = starting_betas.n_cols;
  arma::ivec accept_rec_vec(n_feature);
  accept_rec_vec.zeros();
  betas.zeros();
  //betas.randn();
  betas.slice(0).cols(0, n_beta_start - 1) = starting_betas;
  rhos.row(0) = arma::exp(prior_mean_log_rs.t());
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);

  sigma2.ones();
  //sigma2.row(0) += 9;

  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    // Rcpp::Rcout << "iteration "  << i << std::endl << std::endl;
    betas.slice(i) = para_update_betas_wls_mm(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, accept_rec_vec, grain_size);
    betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
    rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  // Return list with posterior samples
  if(!return_all_re){
    arma::cube betas_sub = betas.tube(arma::span(), arma::span(0, n_beta + n_re_return - 1));
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_sub,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec);
  }


}

//' Negative Binomial GLMM MCMC WLS Force(title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for fixed effects (n_feature x n_beta)
//' @param return_all_re logical variable to determine if posterior samples are returned for random effects (defaults to TRUE)
//' @param n_re_return number of random effects to return a full posterior sample for (defaults to 1, only used if return_all_re = FALSE)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_force(arma::mat counts,
                                       arma::mat design_mat,
                                       arma::mat design_mat_re,
                                       double prior_sd_betas,
                                       double prior_sd_betas_a,
                                       double prior_sd_betas_b,
                                       double prior_sd_rs,
                                       arma::vec prior_mean_log_rs,
                                       int n_it,
                                       double rw_sd_rs,
                                       arma::vec log_offset,
                                       arma::mat starting_betas,
                                       bool return_all_re = true,
                                       int n_re_return = 1,
                                       int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols, n_beta_tot = n_beta + n_beta_re;
  arma::cube betas(n_feature, n_beta_tot, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  double n_beta_start = starting_betas.n_cols;
  arma::ivec accept_rec_vec(n_feature), inv_errors_vec(n_feature);
  accept_rec_vec.zeros();
  inv_errors_vec.zeros();
  betas.zeros();
  //betas.randn();
  betas.slice(0).cols(0, n_beta_start - 1) = starting_betas;
  rhos.row(0) = arma::exp(prior_mean_log_rs.t());
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);

  sigma2.ones();
  //sigma2.row(0) += 9;

  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    //Rcpp::Rcout << "iteration "  << i << std::endl << std::endl;
    betas.slice(i) = para_update_betas_wls_mm_force(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, accept_rec_vec, i, inv_errors_vec, grain_size);
    betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
    rhos.row(i) = arma::trans(para_update_rhos_force(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, i, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  arma::uvec idx_na = arma::find(inv_errors_vec > 0);
  int n_na = idx_na.n_elem;
  if(n_na > 0){
    // arma::vec na_tmp(n_na);
    // na_tmp.fill(NA_REAL);
    // accept_vec.rows(idx_na) = na_tmp;
    accept_rec_vec.elem(idx_na).fill(NA_INTEGER);
    inv_errors_vec.elem(idx_na).fill(1);
    for(int k = 0; k < n_na; k++){
      betas.tube(arma::span(idx_na(k), idx_na(k)), arma::span()).fill(NA_REAL);
    }
    sigma2.cols(idx_na).fill(NA_REAL);
    rhos.cols(idx_na).fill(NA_REAL);
  }

  // arma::vec inv_errors_ret(counts.n_rows);
  // inv_errors_ret = ret(0, 4).slice(0).col(0);
  // Return list with posterior samples
  if(!return_all_re){
    arma::cube betas_sub = betas.tube(arma::span(), arma::span(0, n_beta + n_re_return - 1));
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_sub,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec,
                              Rcpp::Named("inv_errors") = inv_errors_vec);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec,
                              Rcpp::Named("inv_errors") = inv_errors_vec);
  }


}

//' Negative Binomial GLMM MCMC WLS Gamma Dispersion(title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_shape vector of prior gamma shape parameters for dispersions
//' @param prior_scale vector of prior gamma scale parameters for dispersions
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for fixed effects (n_feature x n_beta)
//' @param return_all_re logical variable to determine if posterior samples are returned for random effects (defaults to TRUE)
//' @param n_re_return number of random effects to return a full posterior sample for (defaults to 1, only used if return_all_re = FALSE)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_gam(arma::mat counts,
                                     arma::mat design_mat,
                                     arma::mat design_mat_re,
                                     double prior_sd_betas,
                                     double prior_sd_betas_a,
                                     double prior_sd_betas_b,
                                     arma::vec prior_shape,
                                     arma::vec prior_scale,
                                     int n_it,
                                     double rw_sd_rs,
                                     arma::vec log_offset,
                                     arma::mat starting_betas,
                                     arma::vec starting_disps,
                                     bool return_all_re = true,
                                     int n_re_return = 1,
                                     int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols, n_beta_tot = n_beta + n_beta_re;
  arma::cube betas(n_feature, n_beta_tot, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  double n_beta_start = starting_betas.n_cols;
  arma::ivec accept_rec_vec(n_feature);
  accept_rec_vec.zeros();
  betas.zeros();
  //betas.randn();
  betas.slice(0).cols(0, n_beta_start - 1) = starting_betas;
  //rhos.row(0) = arma::exp(prior_mean_log_rs.t());
  rhos.row(0) = starting_disps.t();
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);


  sigma2.ones();
  //sigma2.row(0) += 9;

  //mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_wls_mm(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, accept_rec_vec, grain_size);
    betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
    //rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, grain_size));
    rhos.row(i) = arma::trans(para_update_rhos_gam(betas.slice(i), counts, rhos.row(i-1).t(), log_offset, design_mat, prior_shape, prior_scale, rw_sd_rs, n_beta_tot, n_sample, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  // Return list with posterior samples
  if(!return_all_re){
    arma::cube betas_sub = betas.tube(arma::span(), arma::span(0, n_beta + n_re_return - 1));
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_sub,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2,
                              Rcpp::Named("accepts") = accept_rec_vec);
  }


}



//' Negative Binomial GLMM MCMC WLS Split (title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for fixed effects (n_feature x n_beta)
//' @param return_all_re logical variable to determine if posterior samples are returned for random effects (defaults to TRUE)
//' @param n_re_return number of random effects to return a full posterior sample for (defaults to 1, only used if return_all_re = FALSE)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_split(arma::mat counts,
                                       arma::mat design_mat,
                                       arma::mat design_mat_re,
                                       double prior_sd_betas,
                                       double prior_sd_betas_a,
                                       double prior_sd_betas_b,
                                       double prior_sd_rs,
                                       arma::vec prior_mean_log_rs,
                                       int n_it,
                                       double rw_sd_rs,
                                       arma::vec log_offset,
                                       arma::mat starting_betas,
                                       bool return_all_re = true,
                                       int n_re_return = 1,
                                       int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols, n_beta_tot = n_beta + n_beta_re;
  arma::cube betas(n_feature, n_beta_tot, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  betas.zeros();
  betas.slice(0).cols(0, n_beta - 1) = starting_betas;
  rhos.row(0) = arma::exp(prior_mean_log_rs.t());
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);

  sigma2.ones();

  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_wls_mm_split(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, grain_size);
    betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
    rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  // Return list with posterior samples
  if(!return_all_re){
    arma::cube betas_sub = betas.tube(arma::span(), arma::span(0, n_beta + n_re_return - 1));
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_sub,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2);
  }


}

//' Negative Binomial GLMM MCMC WLS Split Half (title)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
//' @param prior_sd_rs prior std. dev for dispersion parameters
//' @param prior_mean_log_rs vector of prior means for dispersion parameters
//' @param n_it number of iterations to run MCMC
//' @param rw_sd_rs random wal std. dev. for proposing dispersion values
//' @param log_offset vector of offsets on log scale
//' @param starting_betas matrix of starting values for fixed effects (n_feature x n_beta)
//' @param return_all_re logical variable to determine if posterior samples are returned for random effects (defaults to TRUE)
//' @param n_re_return number of random effects to return a full posterior sample for (defaults to 1, only used if return_all_re = FALSE)
//' @param grain_size minimum size of parallel jobs, defaults to 1, can ignore for now
//'
//' @author Brian Vestal
//'
//' @return
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_split_half(arma::mat counts,
                                            arma::mat design_mat,
                                            arma::mat design_mat_re,
                                            double prior_sd_betas,
                                            double prior_sd_betas_a,
                                            double prior_sd_betas_b,
                                            double prior_sd_rs,
                                            arma::vec prior_mean_log_rs,
                                            int n_it,
                                            double rw_sd_rs,
                                            arma::vec log_offset,
                                            arma::mat starting_betas,
                                            bool return_all_re = true,
                                            int n_re_return = 1,
                                            int grain_size = 1){
  int i, j, n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_feature = counts.n_rows,
    n_sample = counts.n_cols, n_beta_tot = n_beta + n_beta_re;
  arma::cube betas(n_feature, n_beta_tot, n_it);
  arma::mat rhos(n_it, n_feature), betas_cur_mat(n_feature, n_beta_re), sigma2(n_it, n_feature);
  arma::vec beta_cur(n_beta_re), beta_prop(n_feature), mean_cur(n_sample), mean_prop(n_sample), mean_rho_cur(n_feature);
  double  a_rand_int_post, b_rand_int_post;
  betas.zeros();
  betas.slice(0).cols(0, n_beta - 1) = starting_betas;
  rhos.row(0) = arma::exp(prior_mean_log_rs.t());
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);

  sigma2.ones();

  mean_rho_cur = prior_mean_log_rs;

  for(i = 1; i < n_it; i++){
    betas.slice(i) = para_update_betas_wls_mm_split_half(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, grain_size);
    betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
    rhos.row(i) = arma::trans(para_update_rhos(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, grain_size));

    //  Updating random intercept variance
    for(j = 0; j < n_feature; j++){
      beta_cur = betas_cur_mat.row(j).t();
      a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
      b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
      sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    }

  }

  // Return list with posterior samples
  if(!return_all_re){
    arma::cube betas_sub = betas.tube(arma::span(), arma::span(0, n_beta + n_re_return - 1));
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_sub,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2);
  }
  else{
    return Rcpp::List::create(Rcpp::Named("betas_sample") = betas,
                              Rcpp::Named("alphas_sample") = rhos,
                              Rcpp::Named("sigma2_sample") = sigma2);
  }


}

// for(i = 1; i < n_it; i++){
//   //Rcpp::Rcout << "iteration "  << i << std::endl << std::endl;
//   betas.slice(i) = para_update_betas_wls_mm_force(betas.slice(i-1), counts, rhos.row(i-1).t(), log_offset, design_mat_tot, prior_sd_betas, sigma2.row(i-1), n_beta, n_beta_re, n_sample, accept_rec_vec, i, grain_size);
//   betas_cur_mat = betas.slice(i).cols(n_beta, n_beta_tot - 1);
//   rhos.row(i) = arma::trans(para_update_rhos_force(betas.slice(i), counts, rhos.row(i-1).t(), mean_rho_cur, log_offset, design_mat_tot, prior_sd_rs, rw_sd_rs, n_beta_tot, n_sample, i, grain_size));
//
//   //  Updating random intercept variance
//   for(j = 0; j < n_feature; j++){
//     beta_cur = betas_cur_mat.row(j).t();
//     a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0;
//     b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur, beta_cur) / 2.0;
//     sigma2(i, j) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
//   }
//
// }

//   Function to run an entire chain for one feature
arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
                                          const arma::vec &log_offset,
                                          const arma::rowvec &starting_betas,
                                          const arma::mat &design_mat,
                                          const double &mean_rho,
                                          const double &prior_sd_rs,
                                          const double &rw_sd_rs,
                                          const double &prior_sd_betas,
                                          const double &n_beta,
                                          const double &n_beta_re,
                                          const double &n_sample,
                                          const double &prior_sd_betas_a,
                                          const double &prior_sd_betas_b,
                                          const int &n_it){
  int n_beta_tot = n_beta + n_beta_re, i, accepts = 0, inv_errors = 0;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::field<arma::mat> ret(1, 5);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  // Rcpp::Rcout << "Line 2933 check" << std::endl;
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;



  // for(i = 1; i < n_it; i++){
  //   betas_cur = arma::trans(update_betas_wls_mm_force(betas_last,
  //                                                     counts,
  //                                                     disp_sample(i-1),
  //                                                     log_offset,
  //                                                     design_mat,
  //                                                     prior_sd_betas,
  //                                                     sigma2_sample(i-1),
  //                                                     n_beta,
  //                                                     n_beta_re,
  //                                                     n_sample,
  //                                                     accepts,
  //                                                     i,
  //                                                     inv_errors));
  //   betas_last = betas_cur;
  //   beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
  //   betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);
  //   disp_sample(i) = update_rho_force(betas_cur,
  //               counts,
  //               disp_sample(i-1),
  //               mean_rho,
  //               log_offset,
  //               design_mat,
  //               prior_sd_rs,
  //               rw_sd_rs,
  //               n_beta_tot,
  //               n_sample,
  //               i);
  //
  //   b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
  //   sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
  //   if(inv_errors > 0){
  //     betas_sample.fill(NA_REAL);
  //     disp_sample.fill(NA_REAL);
  //     sigma2_sample.fill(NA_REAL);
  //     accepts = -1;
  //     break;
  //   }
  // }
  i = 1;
  while(i < n_it && inv_errors > -1){
    betas_cur = arma::trans(update_betas_wls_mm_force(betas_last,
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
                                                      inv_errors));
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
                i);

    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    i++;
  }

  // if(inv_errors > 0){
  //   betas_sample.fill(NA_REAL);
  //   disp_sample.fill(NA_REAL);
  //   sigma2_sample.fill(NA_REAL);
  //   accepts = -1;
  // }

  ret(0, 0) = betas_sample;
  ret(0, 1) = disp_sample;
  ret(0, 2) = sigma2_sample;
  ret(0, 3) = accepts;
  ret(0, 4) = inv_errors;
  // Rcpp::Rcout << "Line 3024 check" << std::endl;
  return(ret);
}


// arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
//                                           const arma::vec &log_offset,
//                                           const arma::vec &starting_betas,
//                                           const arma::mat &design_mat,
//                                           const double &mean_rho,
//                                           const double &prior_sd_rs,
//                                           const double &rw_sd_rs,
//                                           const double &prior_sd_betas,
//                                           const double &n_beta,
//                                           const double &n_beta_re,
//                                           const double &n_sample,
//                                           const double &prior_sd_betas_a,
//                                           const double &prior_sd_betas_b,
//                                           const int n_it){
struct whole_feature_sample_struct : public Worker
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
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;

  // density that I have accumulated
  arma::cube &upd_betas;
  arma::cube &upd_alphas;
  arma::field<arma::mat> mcmc_res_tmp;
  arma::cube &upd_sigmas;
  arma::cube &upd_accepts;
  arma::cube &upd_inv_errors;

  // constructors
  whole_feature_sample_struct(const arma::mat &counts,
                              const arma::vec &log_offset,
                              const arma::mat &starting_betas,
                              const arma::mat &design_mat,
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
                              arma::cube &upd_betas,
                              arma::cube &upd_alphas,
                              arma::cube &upd_sigmas,
                              arma::cube &upd_accepts,
                              arma::cube &upd_inv_errors)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample), prior_sd_betas_a(prior_sd_betas_a),
      prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), upd_betas(upd_betas), upd_alphas(upd_alphas),
      upd_sigmas(upd_sigmas), upd_accepts(upd_accepts), upd_inv_errors(upd_inv_errors){}

  // arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
  //                                           const arma::vec &log_offset,
  //                                           const arma::vec &starting_betas,
  //                                           const arma::mat &design_mat,
  //                                           const double &mean_rho,
  //                                           const double &prior_sd_rs,
  //                                           const double &rw_sd_rs,
  //                                           const double &prior_sd_betas,
  //                                           const double &n_beta,
  //                                           const double &n_beta_re,
  //                                           const double &n_sample,
  //                                           const double &prior_sd_betas_a,
  //                                           const double &prior_sd_betas_b,
  //                                           const int n_it){
  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      mcmc_res_tmp = whole_chain_nbglmm(counts.row(i),
                                        log_offset,
                                        starting_betas.row(i),
                                        design_mat,
                                        mean_rhos(i),
                                        prior_sd_rs,
                                        rw_sd_rs,
                                        prior_sd_betas,
                                        n_beta,
                                        n_beta_re,
                                        n_sample,
                                        prior_sd_betas_a,
                                        prior_sd_betas_b,
                                        n_it);
      // Rcpp::Rcout << "Line 3124 check" << std::endl;
      upd_betas.slice(i) = mcmc_res_tmp(0, 0);
      // Rcpp::Rcout << "Line 3126 check" << std::endl;
      upd_alphas.slice(0).col(i) = mcmc_res_tmp(0, 1);
      // Rcpp::Rcout << "Line 3128 check" << std::endl;
      upd_sigmas.slice(0).col(i) = mcmc_res_tmp(0, 2);
      // Rcpp::Rcout << "Line 3130 check" << std::endl;
      upd_accepts(i, 0, 0) = mcmc_res_tmp(0, 3)(0, 0);
      // Rcpp::Rcout << "Line 3132 check" << std::endl;
      upd_inv_errors(i, 0, 0) = mcmc_res_tmp(0, 4)(0, 0);
      mcmc_res_tmp.reset();
    }
  }

};

arma::field<arma::cube> mcmc_chain_par(const arma::mat &counts,
                                       const arma::vec &log_offset,
                                       const arma::mat &starting_betas,
                                       const arma::mat &design_mat,
                                       const arma::vec &mean_rhos,
                                       const double &prior_sd_rs,
                                       const double &rw_sd_rs,
                                       const double &prior_sd_betas,
                                       const int &n_beta,
                                       const int &n_beta_re,
                                       const int &n_sample,
                                       const double &prior_sd_betas_a,
                                       const double &prior_sd_betas_b,
                                       const int &n_it){
  arma::cube upd_betas(n_it, n_beta, counts.n_rows, arma::fill::zeros	);
  arma::cube upd_alphas(n_it, counts.n_rows, 1, arma::fill::zeros);
  arma::cube upd_sigmas(n_it, counts.n_rows, 1, arma::fill::zeros);
  arma::cube upd_accepts(counts.n_rows, 1, 1, arma::fill::zeros);
  arma::cube upd_inv_errors(counts.n_rows, 1, 1, arma::fill::zeros);
  arma::field<arma::cube> ret(1, 5);

  whole_feature_sample_struct mcmc_inst(counts,
                                        log_offset,
                                        starting_betas,
                                        design_mat,
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
                                        upd_betas,
                                        upd_alphas,
                                        upd_sigmas,
                                        upd_accepts,
                                        upd_inv_errors);
  parallelFor(0, counts.n_rows, mcmc_inst);
  ret(0, 0) = upd_betas;
  ret(0, 1) = upd_alphas;
  ret(0, 2) = upd_sigmas;
  ret(0, 3) = upd_accepts;
  ret(0, 4) = upd_inv_errors;
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(ret);
}

//' Negative Binomial GLMM MCMC WLS Force (full parallel chians)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
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
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_force_fp(arma::mat counts,
                                          arma::mat design_mat,
                                          arma::mat design_mat_re,
                                          double prior_sd_betas,
                                          double prior_sd_betas_a,
                                          double prior_sd_betas_b,
                                          double prior_sd_rs,
                                          arma::vec prior_mean_log_rs,
                                          int n_it,
                                          double rw_sd_rs,
                                          arma::vec log_offset,
                                          arma::mat starting_betas,
                                          int grain_size = 1){

  arma::field<arma::cube> ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par(counts,
                       log_offset,
                       starting_betas2,
                       design_mat_tot,
                       prior_mean_log_rs,
                       prior_sd_rs,
                       rw_sd_rs,
                       prior_sd_betas,
                       n_beta,
                       n_beta_re,
                       n_sample,
                       prior_sd_betas_a,
                       prior_sd_betas_b,
                       n_it);
  // arma::field<arma::cube> mcmc_chain_par(const arma::mat &counts,
  //                                        const arma::vec &log_offset,
  //                                        const arma::mat &starting_betas,
  //                                        const arma::mat &design_mat,
  //                                        const arma::vec &mean_rhos,
  //                                        const double &prior_sd_rs,
  //                                        const double &rw_sd_rs,
  //                                        const double &prior_sd_betas,
  //                                        const int &n_beta,
  //                                        const int &n_beta_re,
  //                                        const int &n_sample,
  //                                        const double &prior_sd_betas_a,
  //                                        const double &prior_sd_betas_b,
  //                                        const int &n_it){
  arma::vec accept_vec = ret(0, 3).slice(0).col(0);
  arma::uvec idx_na = arma::find(accept_vec < 0);
  int n_na = idx_na.n_elem;
  if(n_na > 0){
    arma::vec na_tmp(n_na);
    na_tmp.fill(NA_REAL);
    accept_vec.rows(idx_na) = na_tmp;
  }

  // arma::vec inv_errors_ret(counts.n_rows);
  // inv_errors_ret = ret(0, 4).slice(0).col(0);

  return Rcpp::List::create(Rcpp::Named("betas_sample") = ret(0, 0),
                            Rcpp::Named("alphas_sample") = ret(0, 1).slice(0),
                            Rcpp::Named("sigma2_sample") = ret(0, 2).slice(0),
                            Rcpp::Named("accepts") = accept_vec,
                            Rcpp::Named("inv_errors") = ret(0, 4).slice(0));
}

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


//   Function to run an entire chain for one feature
arma::field<arma::mat> whole_chain_nbglmm_rw(const arma::rowvec &counts,
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
  int n_beta_tot = n_beta + n_beta_re, accepts = 0, i;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::field<arma::mat> ret(1, 4);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  // Rcpp::Rcout << "Line 2933 check" << std::endl;

  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;



  for(i = 1; i < n_it; i++){
    // Rcpp::Rcout << "Line 2954 check" << std::endl;
    // Rcpp::Rcout << "betas_last = " << betas_last << std::endl;
    // Rcpp::Rcout << "counts = " << counts << std::endl;
    // Rcpp::Rcout << "disp = " << disp_sample(i-1) << std::endl;
    // Rcpp::Rcout << "log_offset = " << log_offset << std::endl;
    // Rcpp::Rcout << "design = " << design_mat.n_cols << std::endl;
    // Rcpp::Rcout << "n_beta = " << n_beta << std::endl;
    // Rcpp::Rcout << "n_beta_re = " << n_beta_re << std::endl;

    // arma::vec update_betas_wls_mm_rw(const arma::rowvec &beta_cur,
    //                                  const arma::rowvec &counts,
    //                                  const double &alpha_cur,
    //                                  const arma::vec &log_offset,
    //                                  const arma::mat &design_mat,
    //                                  const double &prior_sd_betas,
    //                                  const double &rw_var_betas,
    //                                  const double &re_var,
    //                                  const int &n_beta,
    //                                  const int &n_beta_re,
    //                                  const int &n_sample,
    //                                  int &accept_rec,
    //                                  const int &it_num){
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
    // Rcpp::Rcout << "Line 2966 check" << std::endl;
    betas_last = betas_cur;
    beta_cur_re = betas_cur.cols(n_beta, n_beta_tot - 1);
    betas_sample.row(i) = betas_cur.cols(0, n_beta - 1);

    // Rcpp::Rcout << "Line 2971 check" << std::endl;
    // double update_rho_force(const arma::rowvec &beta_cur,
    //                         const arma::rowvec &counts,
    //                         const double &alpha_cur,
    //                         const double &mean_alpha_cur,
    //                         const arma::vec &log_offset,
    //                         const arma::mat &design_mat,
    //                         const double &prior_sd_rs,
    //                         const double &rw_sd_rs,
    //                         const int &n_beta,
    //                         const int &n_sample,
    //                         const int &it_num){
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
                i);
    // Rcpp::Rcout << "Line 2994 check" << std::endl;
    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    // Rcpp::Rcout << "Line 3014 check" << std::endl;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    // Rcpp::Rcout << "Line 3016 check" << std::endl;
  }

  ret(0, 0) = betas_sample;
  ret(0, 1) = disp_sample;
  ret(0, 2) = sigma2_sample;
  arma::mat accepts_mat(1, 1);
  accepts_mat(0, 0) = accepts;
  ret(0, 3) = accepts_mat;
  // Rcpp::Rcout << "Line 3024 check" << std::endl;
  return(ret);
}


struct whole_feature_sample_rw_struct : public Worker
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

  // density that I have accumulated
  arma::cube &upd_betas;
  arma::cube &upd_alphas;
  arma::field<arma::mat> mcmc_res_tmp;
  arma::cube &upd_sigmas;
  arma::cube &upd_accepts;

  // constructors
  whole_feature_sample_rw_struct(const arma::mat &counts,
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
                                 arma::cube &upd_betas,
                                 arma::cube &upd_alphas,
                                 arma::cube &upd_sigmas,
                                 arma::cube &upd_accepts)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      rw_var_betas(rw_var_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample),
      prior_sd_betas_a(prior_sd_betas_a), prior_sd_betas_b(prior_sd_betas_b), n_it(n_it),
      upd_betas(upd_betas), upd_alphas(upd_alphas), upd_sigmas(upd_sigmas), upd_accepts(upd_accepts){}

  // arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
  //                                           const arma::vec &log_offset,
  //                                           const arma::vec &starting_betas,
  //                                           const arma::mat &design_mat,
  //                                           const double &mean_rho,
  //                                           const double &prior_sd_rs,
  //                                           const double &rw_sd_rs,
  //                                           const double &prior_sd_betas,
  //                                           const double &n_beta,
  //                                           const double &n_beta_re,
  //                                           const double &n_sample,
  //                                           const double &prior_sd_betas_a,
  //                                           const double &prior_sd_betas_b,
  //                                           const int n_it){
  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      mcmc_res_tmp = whole_chain_nbglmm_rw(counts.row(i),
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
      // Rcpp::Rcout << "Line 3124 check" << std::endl;
      upd_betas.slice(i) = mcmc_res_tmp(0, 0);
      // Rcpp::Rcout << "Line 3126 check" << std::endl;
      upd_alphas.slice(0).col(i) = mcmc_res_tmp(0, 1);
      // Rcpp::Rcout << "Line 3128 check" << std::endl;
      upd_sigmas.slice(0).col(i) = mcmc_res_tmp(0, 2);
      // Rcpp::Rcout << "Line 3130 check" << std::endl;
      upd_accepts(i, 0, 0) = mcmc_res_tmp(0, 3)(0, 0);
      // Rcpp::Rcout << "Line 3132 check" << std::endl;
    }
  }

};

arma::field<arma::cube> mcmc_chain_rw_par(const arma::mat &counts,
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
  arma::cube upd_betas(n_it, n_beta, counts.n_rows);
  arma::cube upd_alphas(n_it, counts.n_rows, 1);
  arma::cube upd_sigmas(n_it, counts.n_rows, 1);
  arma::cube upd_accepts(counts.n_rows, 1, 1);
  arma::field<arma::cube> ret(1, 4);

  whole_feature_sample_rw_struct mcmc_inst(counts,
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
                                           upd_betas,
                                           upd_alphas,
                                           upd_sigmas,
                                           upd_accepts);
  parallelFor(0, counts.n_rows, mcmc_inst);
  ret(0, 0) = upd_betas;
  ret(0, 1) = upd_alphas;
  ret(0, 2) = upd_sigmas;
  ret(0, 3) = upd_accepts;
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(ret);
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

Rcpp::List nbmm_mcmc_sampler_rw(arma::mat counts,
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

  arma::field<arma::cube> ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  double rw_var_betas = pow(rw_sd_betas, 2);
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_rw_par(counts,
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
  // arma::field<arma::cube> mcmc_chain_rw_par(const arma::mat &counts,
  //                                           const arma::vec &log_offset,
  //                                           const arma::mat &starting_betas,
  //                                           const arma::mat &design_mat,
  //                                           const arma::vec &mean_rhos,
  //                                           const double &prior_sd_rs,
  //                                           const double &rw_sd_rs,
  //                                           const double &rw_var_betas,
  //                                           const double &prior_sd_betas,
  //                                           const int &n_beta,
  //                                           const int &n_beta_re,
  //                                           const int &n_sample,
  //                                           const double &prior_sd_betas_a,
  //                                           const double &prior_sd_betas_b,
  //                                           const int &n_it){
  return Rcpp::List::create(Rcpp::Named("betas_sample") = ret(0, 0),
                            Rcpp::Named("alphas_sample") = ret(0, 1).slice(0),
                            Rcpp::Named("sigma2_sample") = ret(0, 2).slice(0),
                            Rcpp::Named("accepts") = ret(0, 3).slice(0));
}



/*
 * Now a hybrid version that does some random walks before switching to WLS
 */

//   Function to run an entire chain for one feature
arma::field<arma::mat> whole_chain_nbglmm_hybrid(const arma::rowvec &counts,
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
  int n_beta_tot = n_beta + n_beta_re, accepts = 0, i, inv_errors = 0;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::field<arma::mat> ret(1, 5);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  // Rcpp::Rcout << "Line 2933 check" << std::endl;

  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;



  for(i = 1; i < 25; i++){
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
    // Rcpp::Rcout << "Line 2966 check" << std::endl;
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
                i);

    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;

    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));

  }

  for(i = 25; i < n_it; i++){
    betas_cur = arma::trans(update_betas_wls_mm_force(betas_last,
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
                                                      inv_errors));
    // Rcpp::Rcout << "Line 2966 check" << std::endl;
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
                i);
    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
  }

  ret(0, 0) = betas_sample;
  ret(0, 1) = disp_sample;
  ret(0, 2) = sigma2_sample;
  arma::mat accepts_mat(1, 1), inv_mat(1, 1);
  accepts_mat(0, 0) = accepts;
  inv_mat(0, 0) = inv_errors;
  ret(0, 3) = accepts_mat;
  ret(0, 4) = inv_mat;
  return(ret);
}

struct whole_feature_sample_hybrid_struct : public Worker
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

  // density that I have accumulated
  arma::cube &upd_betas;
  arma::cube &upd_alphas;
  arma::field<arma::mat> mcmc_res_tmp;
  arma::cube &upd_sigmas;
  arma::cube &upd_accepts;
  arma::cube &upd_inv_errors;

  // constructors
  whole_feature_sample_hybrid_struct(const arma::mat &counts,
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
                                     arma::cube &upd_betas,
                                     arma::cube &upd_alphas,
                                     arma::cube &upd_sigmas,
                                     arma::cube &upd_accepts,
                                     arma::cube &upd_inv_errors)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      rw_var_betas(rw_var_betas), n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample), prior_sd_betas_a(prior_sd_betas_a),
      prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), upd_betas(upd_betas), upd_alphas(upd_alphas),
      upd_sigmas(upd_sigmas), upd_accepts(upd_accepts), upd_inv_errors(upd_inv_errors){}

  // arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
  //                                           const arma::vec &log_offset,
  //                                           const arma::vec &starting_betas,
  //                                           const arma::mat &design_mat,
  //                                           const double &mean_rho,
  //                                           const double &prior_sd_rs,
  //                                           const double &rw_sd_rs,
  //                                           const double &prior_sd_betas,
  //                                           const double &n_beta,
  //                                           const double &n_beta_re,
  //                                           const double &n_sample,
  //                                           const double &prior_sd_betas_a,
  //                                           const double &prior_sd_betas_b,
  //                                           const int n_it){
  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      mcmc_res_tmp = whole_chain_nbglmm_hybrid(counts.row(i),
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
      // Rcpp::Rcout << "Line 3124 check" << std::endl;
      upd_betas.slice(i) = mcmc_res_tmp(0, 0);
      // Rcpp::Rcout << "Line 3126 check" << std::endl;
      upd_alphas.slice(0).col(i) = mcmc_res_tmp(0, 1);
      // Rcpp::Rcout << "Line 3128 check" << std::endl;
      upd_sigmas.slice(0).col(i) = mcmc_res_tmp(0, 2);
      // Rcpp::Rcout << "Line 3130 check" << std::endl;
      upd_accepts(i, 0, 0) = mcmc_res_tmp(0, 3)(0, 0);
      // Rcpp::Rcout << "Line 3132 check" << std::endl;
      upd_inv_errors(i, 0, 0) = mcmc_res_tmp(0, 4)(0, 0);
    }
  }

};

arma::field<arma::cube> mcmc_chain_hybrid_par(const arma::mat &counts,
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
                                              const int &n_it){
  arma::cube upd_betas(n_it, n_beta, counts.n_rows);
  arma::cube upd_alphas(n_it, counts.n_rows, 1);
  arma::cube upd_sigmas(n_it, counts.n_rows, 1);
  arma::cube upd_accepts(counts.n_rows, 1, 1);
  arma::cube upd_inv_errors(counts.n_rows, 1, 1);
  arma::field<arma::cube> ret(1, 5);

  whole_feature_sample_hybrid_struct mcmc_inst(counts,
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
                                               upd_betas,
                                               upd_alphas,
                                               upd_sigmas,
                                               upd_accepts,
                                               upd_inv_errors);
  parallelFor(0, counts.n_rows, mcmc_inst);
  ret(0, 0) = upd_betas;
  ret(0, 1) = upd_alphas;
  ret(0, 2) = upd_sigmas;
  ret(0, 3) = upd_accepts;
  ret(0, 4) = upd_inv_errors;
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(ret);
}

//' Negative Binomial GLMM MCMC WLS Force Hybrid (full parallel chians)
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
//' @param rw_sd_betas random walk std. dev. for regression coefficients
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
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
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_hybrid(arma::mat counts,
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

  arma::field<arma::cube> ret;
  arma::mat design_mat_tot = arma::join_rows(design_mat, design_mat_re);
  int n_beta = design_mat.n_cols, n_beta_re = design_mat_re.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  double rw_var_betas = pow(rw_sd_betas, 2);
  ret = mcmc_chain_hybrid_par(counts,
                              log_offset,
                              starting_betas2,
                              design_mat_tot,
                              prior_mean_log_rs,
                              prior_sd_rs,
                              rw_sd_rs,
                              prior_sd_betas,
                              rw_var_betas,
                              n_beta,
                              n_beta_re,
                              n_sample,
                              prior_sd_betas_a,
                              prior_sd_betas_b,
                              n_it);
  // arma::field<arma::cube> mcmc_chain_par(const arma::mat &counts,
  //                                        const arma::vec &log_offset,
  //                                        const arma::mat &starting_betas,
  //                                        const arma::mat &design_mat,
  //                                        const arma::vec &mean_rhos,
  //                                        const double &prior_sd_rs,
  //                                        const double &rw_sd_rs,
  //                                        const double &prior_sd_betas,
  //                                        const int &n_beta,
  //                                        const int &n_beta_re,
  //                                        const int &n_sample,
  //                                        const double &prior_sd_betas_a,
  //                                        const double &prior_sd_betas_b,
  //                                        const int &n_it){
  return Rcpp::List::create(Rcpp::Named("betas_sample") = ret(0, 0),
                            Rcpp::Named("alphas_sample") = ret(0, 1).slice(0),
                            Rcpp::Named("sigma2_sample") = ret(0, 2).slice(0),
                            Rcpp::Named("accepts") = ret(0, 3).slice(0),
                            Rcpp::Named("inv_errors") = ret(0, 4).slice(0));
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/////     Trying again but without the use of fields     //////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

//   Function to run an entire chain for one feature
arma::mat whole_chain_nbglmm2(const arma::rowvec &counts,
                              const arma::vec &log_offset,
                              const arma::rowvec &starting_betas,
                              const arma::mat &design_mat,
                              const double &mean_rho,
                              const double &prior_sd_rs,
                              const double &rw_sd_rs,
                              const double &prior_sd_betas,
                              const double &n_beta,
                              const double &n_beta_re,
                              const double &n_sample,
                              const double &prior_sd_betas_a,
                              const double &prior_sd_betas_b,
                              const int &n_it){
  int n_beta_tot = n_beta + n_beta_re, i = 1, accepts = 0, inv_errors = 0;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_it, n_beta + 4, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  // Rcpp::Rcout << "Line 2933 check" << std::endl;
  betas_sample.zeros();
  betas_sample.row(0) = starting_betas.cols(0, n_beta - 1);
  disp_sample.zeros();
  disp_sample(0) = exp(mean_rho);
  sigma2_sample.zeros();
  sigma2_sample(0) = 1;
  betas_cur = starting_betas;
  betas_last = starting_betas;

  while(i < n_it && inv_errors < 1){
    betas_cur = arma::trans(update_betas_wls_mm_force(betas_last,
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
                                                      inv_errors));
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
                i);

    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    i++;
  }

  if(inv_errors > 0){
    betas_sample.fill(NA_REAL);
    disp_sample.fill(NA_REAL);
    sigma2_sample.fill(NA_REAL);
    accepts = -1;
  }

  ret.cols(0, n_beta - 1) = betas_sample;
  ret.col(n_beta) = disp_sample;
  ret.col(n_beta + 1) = sigma2_sample;
  ret(0, n_beta + 2) = accepts;
  ret(0, n_beta + 3) = inv_errors;
  // Rcpp::Rcout << "Line 3024 check" << std::endl;
  return(ret);
}


// arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
//                                           const arma::vec &log_offset,
//                                           const arma::vec &starting_betas,
//                                           const arma::mat &design_mat,
//                                           const double &mean_rho,
//                                           const double &prior_sd_rs,
//                                           const double &rw_sd_rs,
//                                           const double &prior_sd_betas,
//                                           const double &n_beta,
//                                           const double &n_beta_re,
//                                           const double &n_sample,
//                                           const double &prior_sd_betas_a,
//                                           const double &prior_sd_betas_b,
//                                           const int n_it){
struct whole_feature_sample_struct2 : public Worker
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
  const int &n_beta;
  const int &n_beta_re;
  const int &n_sample;
  const double &prior_sd_betas_a;
  const double &prior_sd_betas_b;
  const int &n_it;

  // density that I have accumulated
  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct2(const arma::mat &counts,
                               const arma::vec &log_offset,
                               const arma::mat &starting_betas,
                               const arma::mat &design_mat,
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
                               arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      n_beta(n_beta), n_beta_re(n_beta_re), n_sample(n_sample), prior_sd_betas_a(prior_sd_betas_a),
      prior_sd_betas_b(prior_sd_betas_b), n_it(n_it), upd_param(upd_param){}

  // arma::field<arma::mat> whole_chain_nbglmm(const arma::rowvec &counts,
  //                                           const arma::vec &log_offset,
  //                                           const arma::vec &starting_betas,
  //                                           const arma::mat &design_mat,
  //                                           const double &mean_rho,
  //                                           const double &prior_sd_rs,
  //                                           const double &rw_sd_rs,
  //                                           const double &prior_sd_betas,
  //                                           const double &n_beta,
  //                                           const double &n_beta_re,
  //                                           const double &n_sample,
  //                                           const double &prior_sd_betas_a,
  //                                           const double &prior_sd_betas_b,
  //                                           const int n_it){
  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglmm2(counts.row(i),
                      log_offset,
                      starting_betas.row(i),
                      design_mat,
                      mean_rhos(i),
                      prior_sd_rs,
                      rw_sd_rs,
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

arma::cube mcmc_chain_par2(const arma::mat &counts,
                                        const arma::vec &log_offset,
                                        const arma::mat &starting_betas,
                                        const arma::mat &design_mat,
                                        const arma::vec &mean_rhos,
                                        const double &prior_sd_rs,
                                        const double &rw_sd_rs,
                                        const double &prior_sd_betas,
                                        const int &n_beta,
                                        const int &n_beta_re,
                                        const int &n_sample,
                                        const double &prior_sd_betas_a,
                                        const double &prior_sd_betas_b,
                                        const int &n_it){
  arma::cube upd_param(n_it, n_beta + 4, counts.n_rows, arma::fill::zeros);

  whole_feature_sample_struct2 mcmc_inst(counts,
                                        log_offset,
                                        starting_betas,
                                        design_mat,
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
                                        upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
  return(upd_param);
}

//' Negative Binomial GLMM MCMC WLS Force (full parallel chians)
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
//' @param prior_sd_betas_a alpha in inverse gamma prior for random intercept variance
//' @param prior_sd_betas_b beta in inverse gamma prior for random intercept variance
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
//' Returns a list with a cube of regression parameters, including random effects, a matrix of dispersion values, and a matrix of random intercept variances
//'
//' @export
// [[Rcpp::export]]

Rcpp::List nbmm_mcmc_sampler_wls_force_fp2(arma::mat counts,
                                           arma::mat design_mat,
                                           arma::mat design_mat_re,
                                           double prior_sd_betas,
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
  arma::mat starting_betas2(counts.n_rows, n_beta + n_beta_re);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;
  ret = mcmc_chain_par2(counts,
                       log_offset,
                       starting_betas2,
                       design_mat_tot,
                       prior_mean_log_rs,
                       prior_sd_rs,
                       rw_sd_rs,
                       prior_sd_betas,
                       n_beta,
                       n_beta_re,
                       n_sample,
                       prior_sd_betas_a,
                       prior_sd_betas_b,
                       n_it);

  // arma::vec inv_errors_ret(counts.n_rows);
  // inv_errors_ret = ret(0, 4).slice(0).col(0);
  arma::cube betas_ret;
  arma::mat disp_ret, sigma2_ret;
  arma::vec accepts_ret, inv_errors_ret;

  betas_ret = ret.tube(arma::span(), arma::span(0, n_beta-1));
  disp_ret = ret.tube(arma::span(), arma::span(n_beta, n_beta));
  sigma2_ret = ret.tube(arma::span(), arma::span(n_beta+1, n_beta+1));
  accepts_ret = ret.tube(0, n_beta+2);
  inv_errors_ret = ret.tube(0, n_beta+3);

  arma::uvec idx_na = arma::find(inv_errors_ret > 0);
  int n_na = idx_na.n_elem;
  if(n_na > 0){
    accepts_ret.elem(idx_na).fill(NA_REAL);
  }

  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_ret,
                            Rcpp::Named("alphas_sample") = disp_ret,
                            Rcpp::Named("sigma2_sample") = sigma2_ret,
                            Rcpp::Named("accepts") = accepts_ret,
                            Rcpp::Named("inv_errors") = inv_errors_ret);
}


/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

arma::mat whole_chain_nbglm2(const arma::rowvec &counts,
                                         const arma::vec &log_offset,
                                         const arma::rowvec &starting_betas,
                                         const arma::mat &design_mat,
                                         const double &mean_rho,
                                         const double &prior_sd_betas,
                                         const double &prior_sd_rs,
                                         const double &rw_sd_rs,
                                         const double &n_beta,
                                         const double &n_sample,
                                         const int n_it){
  int i, accepts = 0;
  arma::mat ret(n_it, n_beta + 2, arma::fill::zeros);
  arma::mat betas_sample(n_it, n_beta);
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

  for(i = 1; i < n_it; i++){
    betas_cur = arma::trans(update_betas_wls(betas_last,
                                             counts,
                                             disp_sample(i-1),
                                             log_offset,
                                             design_mat,
                                             prior_sd_betas,
                                             n_beta,
                                             n_sample,
                                             R_mat_diag,
                                             accepts));
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
  }
  ret.cols(0, n_beta - 1) = betas_sample;
  ret.col(n_beta) = disp_sample;
  ret(0, n_beta+1) = accepts;
  return(ret);
}

struct whole_feature_sample_struct_glm2 : public Worker
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
  const int &n_beta;
  const int &n_sample;
  const int &n_it;

  // density that I have accumulated
  arma::cube &upd_param;

  // constructors
  whole_feature_sample_struct_glm2(const arma::mat &counts,
                                  const arma::vec &log_offset,
                                  const arma::mat &starting_betas,
                                  const arma::mat &design_mat,
                                  const arma::vec &mean_rhos,
                                  const double &prior_sd_rs,
                                  const double &rw_sd_rs,
                                  const double &prior_sd_betas,
                                  const int &n_beta,
                                  const int &n_sample,
                                  const int &n_it,
                                  arma::cube &upd_param)
    : counts(counts), log_offset(log_offset), starting_betas(starting_betas), design_mat(design_mat),
      mean_rhos(mean_rhos), prior_sd_rs(prior_sd_rs), rw_sd_rs(rw_sd_rs), prior_sd_betas(prior_sd_betas),
      n_beta(n_beta), n_sample(n_sample), n_it(n_it), upd_param(upd_param){}

  // process just the elements of the range I've been asked to
  void operator()(std::size_t begin, std::size_t end) {
    for(int i = begin; i < end; i++){
      upd_param.slice(i) = whole_chain_nbglm2(counts.row(i),
                                       log_offset,
                                       starting_betas.row(i),
                                       design_mat,
                                       mean_rhos(i),
                                       prior_sd_betas,
                                       prior_sd_rs,
                                       rw_sd_rs,
                                       n_beta,
                                       n_sample,
                                       n_it);
    }
  }

};

arma::cube mcmc_chain_glm_par2(const arma::mat &counts,
                                           const arma::vec &log_offset,
                                           const arma::mat &starting_betas,
                                           const arma::mat &design_mat,
                                           const arma::vec &mean_rhos,
                                           const double &prior_sd_rs,
                                           const double &rw_sd_rs,
                                           const double &prior_sd_betas,
                                           const int &n_beta,
                                           const int &n_sample,
                                           const int &n_it){
  arma::cube upd_param(n_it, n_beta + 2, counts.n_rows);

  whole_feature_sample_struct_glm2 mcmc_inst(counts,
                                            log_offset,
                                            starting_betas,
                                            design_mat,
                                            mean_rhos,
                                            prior_sd_rs,
                                            rw_sd_rs,
                                            prior_sd_betas,
                                            n_beta,
                                            n_sample,
                                            n_it,
                                            upd_param);
  parallelFor(0, counts.n_rows, mcmc_inst);
  // Rcpp::Rcout << "Line 3183 check" << std::endl;
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

Rcpp::List nbglm_mcmc_fp2(arma::mat counts,
                         arma::mat design_mat,
                         double prior_sd_betas,
                         double prior_sd_rs,
                         arma::vec prior_mean_log_rs,
                         int n_it,
                         double rw_sd_rs,
                         arma::vec log_offset,
                         arma::mat starting_betas,
                         int grain_size = 1){

  arma::cube ret;
  int n_beta = design_mat.n_cols, n_sample = counts.n_cols;
  int n_beta_start = starting_betas.n_cols;
  arma::mat starting_betas2(counts.n_rows, n_beta);
  starting_betas2.zeros();
  starting_betas2.cols(0, n_beta_start - 1) = starting_betas;

  // arma::field<arma::cube> mcmc_chain_glm_par(const arma::mat &counts,
  //                                            const arma::vec &log_offset,
  //                                            const arma::mat &starting_betas,
  //                                            const arma::mat &design_mat,
  //                                            const arma::vec &mean_rhos,
  //                                            const double &prior_sd_rs,
  //                                            const double &rw_sd_rs,
  //                                            const double &prior_sd_betas,
  //                                            const int &n_beta,
  //                                            const int &n_sample,
  //                                            const int &n_it){
  ret = mcmc_chain_glm_par2(counts,
                           log_offset,
                           starting_betas2,
                           design_mat,
                           prior_mean_log_rs,
                           prior_sd_rs,
                           rw_sd_rs,
                           prior_sd_betas,
                           n_beta,
                           n_sample,
                           n_it);


  arma::cube betas_ret;
  arma::mat disp_ret;
  arma::vec accepts_ret;

  betas_ret = ret.tube(arma::span(), arma::span(0, n_beta-1));
  disp_ret = ret.tube(arma::span(), arma::span(n_beta, n_beta));
  accepts_ret = ret.tube(0, n_beta+1);

  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_ret,
                            Rcpp::Named("alphas_sample") = disp_ret,
                            Rcpp::Named("accepts") = accepts_ret);
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
  int n_beta_tot = n_beta + n_beta_re, accepts = 0, i;
  double a_rand_int_post = prior_sd_betas_a + n_beta_re / 2.0, b_rand_int_post;
  arma::mat ret(n_it, n_beta + 3);
  arma::mat betas_sample(n_it, n_beta);
  arma::rowvec betas_cur(n_beta_tot), beta_cur_re(n_beta_re), betas_last(n_beta_tot);
  arma::vec disp_sample(n_it), sigma2_sample(n_it);
  // Rcpp::Rcout << "Line 2933 check" << std::endl;

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
    // Rcpp::Rcout << "Line 2966 check" << std::endl;
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
                i);
    // Rcpp::Rcout << "Line 2994 check" << std::endl;
    b_rand_int_post = prior_sd_betas_b + arma::dot(beta_cur_re.t(), beta_cur_re.t()) / 2.0;
    // Rcpp::Rcout << "Line 3014 check" << std::endl;
    sigma2_sample(i) = 1.0 / (R::rgamma(a_rand_int_post, 1.0 / b_rand_int_post));
    // Rcpp::Rcout << "Line 3016 check" << std::endl;
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

  // density that I have accumulated
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

  return Rcpp::List::create(Rcpp::Named("betas_sample") = betas_ret,
                            Rcpp::Named("alphas_sample") = disp_ret,
                            Rcpp::Named("sigma2_sample") = sigma2_ret,
                            Rcpp::Named("accepts") = accepts_ret);
}

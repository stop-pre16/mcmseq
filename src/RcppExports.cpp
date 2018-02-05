// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// nbglm_mcmc_fp
Rcpp::List nbglm_mcmc_fp(arma::mat counts, arma::mat design_mat, double prior_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbglm_mcmc_fp(SEXP countsSEXP, SEXP design_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_fp(counts, design_mat, prior_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbglm_mcmc
Rcpp::List nbglm_mcmc(arma::mat counts, arma::mat design_mat, double prior_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_betas, double rw_sd_rs, arma::vec log_offset, int grain_size);
RcppExport SEXP _mcmseq_nbglm_mcmc(SEXP countsSEXP, SEXP design_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_betasSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas(rw_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc(counts, design_mat, prior_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_betas, rw_sd_rs, log_offset, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbglm_mcmc_wls
Rcpp::List nbglm_mcmc_wls(arma::mat counts, arma::mat design_mat, double prior_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbglm_mcmc_wls(SEXP countsSEXP, SEXP design_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_wls(counts, design_mat, prior_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbglm_mcmc_wls_gam
Rcpp::List nbglm_mcmc_wls_gam(arma::mat counts, arma::mat design_mat, double prior_sd_betas, arma::vec prior_shape, arma::vec prior_scale, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, arma::vec starting_disps, int grain_size);
RcppExport SEXP _mcmseq_nbglm_mcmc_wls_gam(SEXP countsSEXP, SEXP design_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_shapeSEXP, SEXP prior_scaleSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP starting_dispsSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_shape(prior_shapeSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_scale(prior_scaleSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type starting_disps(starting_dispsSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_wls_gam(counts, design_mat, prior_sd_betas, prior_shape, prior_scale, n_it, rw_sd_rs, log_offset, starting_betas, starting_disps, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler
Rcpp::List nbmm_mcmc_sampler(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_betas, double rw_sd_betas_re, double rw_sd_rs, arma::vec log_offset, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_betasSEXP, SEXP rw_sd_betas_reSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas(rw_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas_re(rw_sd_betas_reSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_betas, rw_sd_betas_re, rw_sd_rs, log_offset, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls
Rcpp::List nbmm_mcmc_sampler_wls(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, bool return_all_re, int n_re_return, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP return_all_reSEXP, SEXP n_re_returnSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< bool >::type return_all_re(return_all_reSEXP);
    Rcpp::traits::input_parameter< int >::type n_re_return(n_re_returnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, return_all_re, n_re_return, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_force
Rcpp::List nbmm_mcmc_sampler_wls_force(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, bool return_all_re, int n_re_return, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_force(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP return_all_reSEXP, SEXP n_re_returnSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< bool >::type return_all_re(return_all_reSEXP);
    Rcpp::traits::input_parameter< int >::type n_re_return(n_re_returnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_force(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, return_all_re, n_re_return, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_gam
Rcpp::List nbmm_mcmc_sampler_wls_gam(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, arma::vec prior_shape, arma::vec prior_scale, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, arma::vec starting_disps, bool return_all_re, int n_re_return, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_gam(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_shapeSEXP, SEXP prior_scaleSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP starting_dispsSEXP, SEXP return_all_reSEXP, SEXP n_re_returnSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_shape(prior_shapeSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_scale(prior_scaleSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type starting_disps(starting_dispsSEXP);
    Rcpp::traits::input_parameter< bool >::type return_all_re(return_all_reSEXP);
    Rcpp::traits::input_parameter< int >::type n_re_return(n_re_returnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_gam(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_shape, prior_scale, n_it, rw_sd_rs, log_offset, starting_betas, starting_disps, return_all_re, n_re_return, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_split
Rcpp::List nbmm_mcmc_sampler_wls_split(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, bool return_all_re, int n_re_return, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_split(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP return_all_reSEXP, SEXP n_re_returnSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< bool >::type return_all_re(return_all_reSEXP);
    Rcpp::traits::input_parameter< int >::type n_re_return(n_re_returnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_split(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, return_all_re, n_re_return, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_split_half
Rcpp::List nbmm_mcmc_sampler_wls_split_half(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, bool return_all_re, int n_re_return, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_split_half(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP return_all_reSEXP, SEXP n_re_returnSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< bool >::type return_all_re(return_all_reSEXP);
    Rcpp::traits::input_parameter< int >::type n_re_return(n_re_returnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_split_half(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, return_all_re, n_re_return, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_force_fp
Rcpp::List nbmm_mcmc_sampler_wls_force_fp(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_force_fp(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_force_fp(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_rw
Rcpp::List nbmm_mcmc_sampler_rw(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double rw_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_rw(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP rw_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas(rw_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_rw(counts, design_mat, design_mat_re, prior_sd_betas, rw_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_hybrid
Rcpp::List nbmm_mcmc_sampler_wls_hybrid(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double rw_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_hybrid(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP rw_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas(rw_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_hybrid(counts, design_mat, design_mat_re, prior_sd_betas, rw_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbmm_mcmc_sampler_wls_force_fp2
Rcpp::List nbmm_mcmc_sampler_wls_force_fp2(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbmm_mcmc_sampler_wls_force_fp2(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbmm_mcmc_sampler_wls_force_fp2(counts, design_mat, design_mat_re, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}
// nbglm_mcmc_fp2
Rcpp::List nbglm_mcmc_fp2(arma::mat counts, arma::mat design_mat, double prior_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, int grain_size);
RcppExport SEXP _mcmseq_nbglm_mcmc_fp2(SEXP countsSEXP, SEXP design_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP grain_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_fp2(counts, design_mat, prior_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, grain_size));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mcmseq_nbglm_mcmc_fp", (DL_FUNC) &_mcmseq_nbglm_mcmc_fp, 10},
    {"_mcmseq_nbglm_mcmc", (DL_FUNC) &_mcmseq_nbglm_mcmc, 10},
    {"_mcmseq_nbglm_mcmc_wls", (DL_FUNC) &_mcmseq_nbglm_mcmc_wls, 10},
    {"_mcmseq_nbglm_mcmc_wls_gam", (DL_FUNC) &_mcmseq_nbglm_mcmc_wls_gam, 11},
    {"_mcmseq_nbmm_mcmc_sampler", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler, 14},
    {"_mcmseq_nbmm_mcmc_sampler_wls", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls, 15},
    {"_mcmseq_nbmm_mcmc_sampler_wls_force", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_force, 15},
    {"_mcmseq_nbmm_mcmc_sampler_wls_gam", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_gam, 16},
    {"_mcmseq_nbmm_mcmc_sampler_wls_split", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_split, 15},
    {"_mcmseq_nbmm_mcmc_sampler_wls_split_half", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_split_half, 15},
    {"_mcmseq_nbmm_mcmc_sampler_wls_force_fp", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_force_fp, 13},
    {"_mcmseq_nbmm_mcmc_sampler_rw", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_rw, 14},
    {"_mcmseq_nbmm_mcmc_sampler_wls_hybrid", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_hybrid, 14},
    {"_mcmseq_nbmm_mcmc_sampler_wls_force_fp2", (DL_FUNC) &_mcmseq_nbmm_mcmc_sampler_wls_force_fp2, 13},
    {"_mcmseq_nbglm_mcmc_fp2", (DL_FUNC) &_mcmseq_nbglm_mcmc_fp2, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_mcmseq(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

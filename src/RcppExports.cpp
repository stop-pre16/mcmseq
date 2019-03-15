// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// nbglm_mcmc_rw
Rcpp::List nbglm_mcmc_rw(arma::mat counts, arma::mat design_mat, arma::mat contrast_mat, double prior_sd_betas, double rw_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, double prop_burn, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglm_mcmc_rw(SEXP countsSEXP, SEXP design_matSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP rw_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burnSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_betas(rw_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn(prop_burnSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_rw(counts, design_mat, contrast_mat, prior_sd_betas, rw_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, prop_burn, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_rw
Rcpp::List nbglmm_mcmc_rw(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double rw_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, double prop_burn, int grain_size, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_rw(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP rw_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burnSEXP, SEXP grain_sizeSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
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
    Rcpp::traits::input_parameter< double >::type prop_burn(prop_burnSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_rw(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, rw_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, prop_burn, grain_size, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglm_mcmc_wls
Rcpp::List nbglm_mcmc_wls(arma::mat counts, arma::mat design_mat, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, double burn_in_prop, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglm_mcmc_wls(SEXP countsSEXP, SEXP design_matSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP burn_in_propSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type burn_in_prop(burn_in_propSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglm_mcmc_wls(counts, design_mat, contrast_mat, prior_sd_betas, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, burn_in_prop, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_wls
Rcpp::List nbglmm_mcmc_wls(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, double prop_burn_in, int grain_size, int num_accept, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_wls(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burn_inSEXP, SEXP grain_sizeSEXP, SEXP num_acceptSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn_in(prop_burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type num_accept(num_acceptSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_wls(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, prop_burn_in, grain_size, num_accept, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_wls2
Rcpp::List nbglmm_mcmc_wls2(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, double rw_sd_sigma, arma::vec log_offset, arma::mat starting_betas, double prop_burn_in, int grain_size, int num_accept, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_wls2(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP rw_sd_sigmaSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burn_inSEXP, SEXP grain_sizeSEXP, SEXP num_acceptSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_sigma(rw_sd_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn_in(prop_burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type num_accept(num_acceptSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_wls2(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, rw_sd_sigma, log_offset, starting_betas, prop_burn_in, grain_size, num_accept, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_wls3
Rcpp::List nbglmm_mcmc_wls3(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, double rw_sd_sigma, arma::vec log_offset, arma::mat starting_betas, double prop_burn_in, double tau, int grain_size, int num_accept, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_wls3(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP rw_sd_sigmaSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burn_inSEXP, SEXP tauSEXP, SEXP grain_sizeSEXP, SEXP num_acceptSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_sigma(rw_sd_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn_in(prop_burn_inSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type num_accept(num_acceptSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_wls3(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, rw_sd_sigma, log_offset, starting_betas, prop_burn_in, tau, grain_size, num_accept, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_wls4
Rcpp::List nbglmm_mcmc_wls4(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, double rw_sd_sigma, arma::vec log_offset, arma::mat starting_betas, double prop_burn_in, double tau, int grain_size, int num_accept, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_wls4(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP rw_sd_sigmaSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burn_inSEXP, SEXP tauSEXP, SEXP grain_sizeSEXP, SEXP num_acceptSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_sigma(rw_sd_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn_in(prop_burn_inSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type num_accept(num_acceptSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_wls4(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, rw_sd_sigma, log_offset, starting_betas, prop_burn_in, tau, grain_size, num_accept, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}
// nbglmm_mcmc_wls_wc
Rcpp::List nbglmm_mcmc_wls_wc(arma::mat counts, arma::mat design_mat, arma::mat design_mat_re, arma::mat contrast_mat, double prior_sd_betas, double prior_sd_betas_a, double prior_sd_betas_b, double prior_sd_rs, arma::vec prior_mean_log_rs, int n_it, double rw_sd_rs, arma::vec log_offset, arma::mat starting_betas, double prop_burn_in, int grain_size, int num_accept, bool return_cont, Rcpp::StringVector beta_names, Rcpp::StringVector cont_names);
RcppExport SEXP _mcmseq_nbglmm_mcmc_wls_wc(SEXP countsSEXP, SEXP design_matSEXP, SEXP design_mat_reSEXP, SEXP contrast_matSEXP, SEXP prior_sd_betasSEXP, SEXP prior_sd_betas_aSEXP, SEXP prior_sd_betas_bSEXP, SEXP prior_sd_rsSEXP, SEXP prior_mean_log_rsSEXP, SEXP n_itSEXP, SEXP rw_sd_rsSEXP, SEXP log_offsetSEXP, SEXP starting_betasSEXP, SEXP prop_burn_inSEXP, SEXP grain_sizeSEXP, SEXP num_acceptSEXP, SEXP return_contSEXP, SEXP beta_namesSEXP, SEXP cont_namesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type counts(countsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat(design_matSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type design_mat_re(design_mat_reSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type contrast_mat(contrast_matSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas(prior_sd_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_a(prior_sd_betas_aSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_betas_b(prior_sd_betas_bSEXP);
    Rcpp::traits::input_parameter< double >::type prior_sd_rs(prior_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prior_mean_log_rs(prior_mean_log_rsSEXP);
    Rcpp::traits::input_parameter< int >::type n_it(n_itSEXP);
    Rcpp::traits::input_parameter< double >::type rw_sd_rs(rw_sd_rsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_offset(log_offsetSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type starting_betas(starting_betasSEXP);
    Rcpp::traits::input_parameter< double >::type prop_burn_in(prop_burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type grain_size(grain_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type num_accept(num_acceptSEXP);
    Rcpp::traits::input_parameter< bool >::type return_cont(return_contSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type beta_names(beta_namesSEXP);
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type cont_names(cont_namesSEXP);
    rcpp_result_gen = Rcpp::wrap(nbglmm_mcmc_wls_wc(counts, design_mat, design_mat_re, contrast_mat, prior_sd_betas, prior_sd_betas_a, prior_sd_betas_b, prior_sd_rs, prior_mean_log_rs, n_it, rw_sd_rs, log_offset, starting_betas, prop_burn_in, grain_size, num_accept, return_cont, beta_names, cont_names));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mcmseq_nbglm_mcmc_rw", (DL_FUNC) &_mcmseq_nbglm_mcmc_rw, 15},
    {"_mcmseq_nbglmm_mcmc_rw", (DL_FUNC) &_mcmseq_nbglmm_mcmc_rw, 19},
    {"_mcmseq_nbglm_mcmc_wls", (DL_FUNC) &_mcmseq_nbglm_mcmc_wls, 14},
    {"_mcmseq_nbglmm_mcmc_wls", (DL_FUNC) &_mcmseq_nbglmm_mcmc_wls, 19},
    {"_mcmseq_nbglmm_mcmc_wls2", (DL_FUNC) &_mcmseq_nbglmm_mcmc_wls2, 20},
    {"_mcmseq_nbglmm_mcmc_wls3", (DL_FUNC) &_mcmseq_nbglmm_mcmc_wls3, 21},
    {"_mcmseq_nbglmm_mcmc_wls4", (DL_FUNC) &_mcmseq_nbglmm_mcmc_wls4, 21},
    {"_mcmseq_nbglmm_mcmc_wls_wc", (DL_FUNC) &_mcmseq_nbglmm_mcmc_wls_wc, 19},
    {NULL, NULL, 0}
};

RcppExport void R_init_mcmseq(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

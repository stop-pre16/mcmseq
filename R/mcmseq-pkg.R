#' @title mcmseq: Bayesian Hierarchical Negative Binomial Mixed Models for RNA-Seq
#'
#' @description
#' Fitting utilities for estimating negative binomial generalized linear mixed models for use
#' with RNA-Seq data from studies with clustered/longitudinal study designs
#'
#'
#' @importFrom Rcpp sourceCpp
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats terms
#' @importFrom stats quantile
#' @importFrom stats median
#' @importFrom stats model.matrix
#' @importFrom stats p.adjust
#' @useDynLib mcmseq
#'
#' @docType package
#' @name mcmseq-pkg
NULL

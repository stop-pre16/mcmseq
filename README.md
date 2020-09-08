# mcmseq
This is the repo for the mcmseq R package, which has the wrapper functions to run the MCMSeq method described in "MCMSeq: Bayesian hierarchical modeling of clustered and repeated measures RNA sequencing experiments".

To install the package, the Rcpp, RcppArmadillo, RcppParallel, and devtools packages need to first be installed:

install.packages(pkgs = c('Rcpp', 'RcppArmadillo', 'RcppParallel', 'devtools'))

Next, the mcmseq package can be installed using devtools:

devtools::install_github("stop-pre16/mcmseq", build_vignettes = T)

To read the detailed vignette with an example analysis, run:

vignette("mcmseq_vignette")

#' Function to Check for Convergence Issues in the MCMC Chains
#'
#' Checks Geweke statistics and acceptance rates for model parameters to determine convergence issues.
#'
#' @param mcmseqModel A model fit with the mcmseq.fit function.
#' @param geweke.p The p-value for the Geweke statistic that should be used to determine convergence.  By default Geweke p-values less than 0.05 for any parameter are considered convergence failures.
#' @param prop.accepts.betas The acceptance rate for regression coefficients below which the model is considered a convergence failure. Default is 0.1.
#' @param prop.accepts.alphas The acceptance rate for dispersion parameters below which the model is considered a convergence failure. Default is 0.1.
#' #'
#' @return A data frame including Geweke statistic and acceptance rates for genes that failed to converge.
#'
#' @examples
#'
#'
#' @export
#'

mcmseq.covergence <- function(mcmseqModel,
                              geweke.p=0.05,
                              prop.accepts.betas = 0.1,
                              prop.accepts.alphas = 0.1){

  failed_geweke <- failed_betas <- failed_alphas <- NULL

  # Check Geweke
  if(is.null(geweke.p)==F) {
    adjusted_geweke <- apply(mcmseqModel$geweke_all, 2, function(x) p.adjust(x, method='BH'))
    colnames(adjusted_geweke) <- paste0(colnames(adjusted_geweke), "_BH_adjusted_geweke_p")
    failed_geweke <- which(rowSums(adjusted_geweke < geweke.p)>0)}

  # Check prop.accepts.betas
  if(is.null(prop.accepts.betas)==F) failed_betas <- which((mcmseqModel$accepts_betas/mcmseqModel$n_it) < prop.accepts.betas)

  # Check accpet_alphas_limit
  if(is.null(prop.accepts.alphas)==F) failed_alphas <- which((mcmseqModel$accepts_alphas/mcmseqModel$n_it) < prop.accepts.alphas)

  failed_to_converge <- unique(c(failed_geweke, failed_betas, failed_alphas))

  if(length(failed_to_converge)==0){print("Models for all genes converged.")
  }else{print(paste0(length(failed_to_converge), " genes did not meet convergence criteria. Check Geweke statistics and acceptance rates to diagnose problems."))}

  return(data.frame(index = failed_to_converge,
                    mcmseqModel$gene_names[failed_to_converge],
                    adjusted_geweke[failed_to_converge, ],
                    accept_rate_betas = (mcmseqModel$accepts_betas[failed_to_converge]/mcmseqModel$n_it),
                    accept_rate_alphas = (mcmseqModel$accepts_alphas[failed_to_converge]/mcmseqModel$n_it)
                    ))

}

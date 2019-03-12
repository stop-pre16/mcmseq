#' Function to Fit MCMSeq Models
#'
#' Fits negative binomial generlized linear models and negative binomial generalized linear mixed models to RNA-Seq data using MCMC.
#'
#' @param counts A (G x N) numeric matrix or data frame of RNA-seq counts, with genes in rows and samples in columns. G = number of genes.  N = number of samples.
#' @param gene_names An optional character vector of gene names (length G).  If unspecified, row names from the counts matrix will be used.
#' @param fixed_effects A one-sided linear formula describing the fixed-effects part of the model. The independent variable terms are listed to the right of the ~ and separated by + operators.
#' @param sample_data Data frame with N rows containing the fixed effects terms included in the fixed_effects formula, as well as any random effects listed in random_intercept.  The rows of the data frame must correspond (and be in the same order as) the columns of the counts matrix.
#' @param random_intercept An optional character string indicating the name of the clustering variable in sample_data to use as a random intercept.
#' @param contrast_mat An optional C x P numeric matrix of linear contrasts to test, where C is the number of contrasts and P is the number of fixed effects parameters in the model.  Each row is a contrast.
#' @param contrast_names An optional character vector of contrast names (length C).  If unspecified, row names from contrast_mat will be used.
#' @param log_offset A vector of offsets (on the linear predictor/natural log scale) to account for differences in sequencing depth between samples.  If no offset is specified, the ln(75th percentile of counts for sample i) / median(ln(75th percentile of counts) across all samples) is used.
#' @param proposal_method Method for updating the regression coefficients.  Either 'rw" (random walk) or 'wls' (weighted least squares).  Default is 'wls'.
#' @param prior_sd_betas Standard deviation, s for the normal(0, s^2) prior for regression coefficients.  Default is 7.
#' @param prior_u_re_var U parameter in inverse gamma prior for random intercept variance.  Default is 0.01.
#' @param prior_v_re_var V parameter in inverse gamma prior for random intercept variance.  Default is 0.01.
#' @param prior_sd_log_alpha Standard deviation in the log-normal prior for dispersion parameters.  Default is 7.
#' @param prior_mean_log_alpha Vector of prior means for log of dispersion parameters.  If a single value is provided (rather than a vector), the same value will be used for all genes. Default is -ln(2) for all genes.
#' @param n_it Number of MCMC iterations.  Default is 30,000.
#' @param rw_sd_log_alpha Standard deviation for the random walk proposal for dispersion values (normal distribution centerd at current value)
#' @param starting_betas Numeric matrix of starting values for the regression coefficients. For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
#' @param rw_sd_betas Standard deviation for the random walk proposal for regression coefficients.  Only used for propsal_method = 'rw'.  Default is 0.5.
#' @param prop_burn_in Number between 0 and 1 indicating the proportion of MCMC chain to discard as burn-in.  Default is 0.1.
#' @param num_accept Number of forced accepts of fixed and random effects at the beginning of the MCMC. In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall.  This value must be less than or equal to the number of burn-in iterations.
#' #'
#' @return An MCMSeq Object
#'
#' @examples
#'
#'
#' @export
#'

mcmseq.fit <- function(counts=NULL, # matrix of RNA-Seq counts where rows are genes and columns are samples
                       gene_names = NULL, # a vector of gene names (the length of the number of rows in the counts matrix).  If unspecified, rownames from the counts matrix will be used.
                       fixed_effects = NULL, #formula for fixed effects
                       sample_data = NULL,
                       random_intercept = NULL, # An optional character string indicating the name of the clustering variable in sample_data to use as a random intercept.
                       contrast_mat = NULL, # an optional n x p matrix of linear contrasts to test, where n is the number of contrasts and p is the number of fixed effects parameters in the model.  Each row is a contrast.
                       contrast_names = NULL,
                       log_offset = NULL, # a vector of offsets (on the linear predictor/ natural log scale) to account for differences in sequencing depth between samples.  If no offset is specified, we use....
                       proposal_method = 'wls', # either 'rw" (random walk) or 'wls' (weighted least squares)
                       prior_sd_betas = 7, # Standard deviation, s for the normal(0, s^2) prior for regression coefficients
                       prior_u_re_var = 0.01, #prior_sd_betas_a, #U parameter in inverse gamma prior for random intercept variance
                       prior_v_re_var = 0.01, #prior_sd_betas_b, #V parameter in inverse gamma prior for random intercept variance
                       prior_sd_log_alpha = 7, #prior_sd_rs,#Prior std. dev in log-normal prior for dispersion parameters
                       prior_mean_log_alpha = NULL, #prior_mean_log_rs, #Vector of prior means for log of dispersion parameters
                       n_it = 1000, # Number of MCMC iterations
                       rw_sd_log_alpha = 0.5, #rw_sd_rs, # Random walk std. dev. for proposing dispersion values (normal distribution centerd at current value)
                       starting_betas = NULL,#Numeric matrix of starting values for the regression coefficients. For best results, supply starting values for at least the intercept (e.g. row means of counts matrix)
                       rw_sd_betas = 0.5, # random walk std. dev. for proposing beta values
                       prop_burn_in = 0.1, #Proportion of MCMC chain to discard as burn-in when computing summaries
                       num_accept = 20L #Number of forced accepts of fixed and random effects at the beginning of the MCMC. In practice forcing about 20 accepts (default value) prevents inverse errors at the start of chains and gives better mixing overall
){

  # For now set grain_size to default; Brian may remove this later.
  grain_size = 1L

  # Determine if contrasts will be run and create a variable to pass info to the C++ functions
  run_contrasts = ifelse(is.null(contrast_mat)==F, TRUE, FALSE)

  ############################################################################################################
  #Error Messages for insufficient or inconsistent information
  ############################################################################################################

  ### Insufficient Information ###

  if(is.null(counts)==T ) {
    stop("A counts matrix must be provided.")}

  if(is.null(fixed_effects)==T ) {
    stop("A fixed_effects formula must be provided.")}

  if(is.null(sample_data)==T ) {
    stop("sample_data is missing.")}

  ### Inconsistent information ###

  fixed_terms <- attributes(terms(fixed_effects))$term.labels[attributes(terms(fixed_effects))$order==1]

  if(sum(fixed_terms %in% colnames(sample_data)) != length(fixed_terms)){
    stop(paste0("The following fixed effects terms are missing from the sample_data:",
                fixed_terms[!(fixed_terms %in% colnames(sample_data))]))}

  if((ncol(counts)==nrow(sample_data))==F ) {
    stop("The counts matrix and sample data include differing numbers of samples.")}

  if(is.null(gene_names)==F & (nrow(counts)==length(gene_names))==F ) {
    print("The counts matrix and gene_names indicate differing numbers of genes.
          Row names of the counts matrix will be used as gene names.")
    gene_names = rownames(counts)
  }

  if(is.null(contrast_names)==F &
     run_contrasts==TRUE){if((nrow(contrast_mat)==length(contrast_names))==F ){
       print("The contrast_mat and contrast_names indicate differing numbers of contrasts.
             Row names of the contrast_mat will be used as contrast_names.")
       contrast_names = rownames(contrast_mat)}}

  if(is.null(log_offset)==F & (ncol(counts)==length(log_offset))==F ) {
    print("The log_offset vector and counts matrix indicate differing numbers of samples.
          The default log_offset will be calculated from the counts matrix.")
    log_offset = log(apply(counts, 2, function(x) quantile(x, probs=0.75)/median(apply(counts,2,quantile, p = 0.75))))
  }

  if(is.null(prior_mean_log_alpha)==F & length(prior_mean_log_alpha)==1 ){
    prior_mean_log_alpha = rep(prior_mean_log_alpha, nrow(counts))
  }

  if(is.null(prior_mean_log_alpha)==F & (nrow(counts)==length(prior_mean_log_alpha))==F ) {
    print("The prior_mean_log_alpha vector and counts matrix indicate differing numbers of genes
          The default prior_mean_log_alpha will be used.")
    prior_mean_log_alpha = rep(-log(2), nrow(counts))
  }

  if ((proposal_method %in% c('wls', 'rw') )==F){
    print("Invalid proposal_method.  wls will be used.")
    proposal_method = 'wls'
  }

  ### Vector supplied instead of single value ###

  if(length(prior_sd_betas)>1){
    print("Only a single value can be specified for prior_sd_betas.  The first value will be used for all genes and coefficients.")
    prior_sd_betas <- prior_sd_betas[1]
  }

  if(length(prior_u_re_var)>1){
    print("Only a single value can be specified for prior_u_re_var.  The first value will be used for all genes.")
    prior_u_re_var <- prior_u_re_var[1]
  }

  if(length(prior_v_re_var)>1){
    print("Only a single value can be specified for prior_v_re_var.  The first value will be used for all genes.")
    prior_v_re_var <- prior_v_re_var[1]
  }

  if(length(prior_sd_log_alpha)>1){
    print("Only a single value can be specified for prior_sd_log_alpha.  The first value will be used for all genes.")
    prior_sd_log_alpha <- prior_sd_log_alpha[1]
  }

  if(length(n_it)>1){
    print("Only a single value can be specified for n_it.  The first value will be used for all genes.")
    n_it <- n_it[1]
  }

  if(length(rw_sd_log_alpha)>1){
    print("Only a single value can be specified for rw_sd_log_alpha.  The first value will be used for all genes.")
    rw_sd_log_alpha <- rw_sd_log_alpha[1]
  }

  if(length(rw_sd_betas)>1){
    print("Only a single value can be specified for rw_sd_betas.  The first value will be used for all genes and coefficients.")
    rw_sd_betas <- rw_sd_betas[1]
  }

  if(length(prop_burn_in)>1){
    print("Only a single value can be specified for prop_burn_in.  The first value will be used for all genes.")
    prop_burn_in <- prop_burn_in[1]
  }

  if(length(num_accept)>1){
    print("Only a single value can be specified for num_accept.  The first value will be used for all genes.")
    num_accept <- num_accept[1]
  }

  ### Out of range values ###
  if((prior_sd_betas)<=0){
    stop("prior_sd_betas must be greater than 0.")
  }

  if((prior_u_re_var)<=0){
    stop("prior_u_re_var must be greater than 0.")
  }

  if((prior_v_re_var)<=0){
    stop("prior_v_re_var must be greater than 0.")
  }

  if((prior_sd_log_alpha)<=0){
    stop("prior_sd_log_alpha must be greater than 0.")
  }

  if((n_it)<=0){
    stop("n_it must be an integer greater than 0.")
  }

  if((rw_sd_log_alpha)<=0){
    stop("rw_sd_log_alpha must be greater than 0.")
  }

  if((rw_sd_betas)<=0){
    stop("rw_sd_betas must be greater than 0.")
  }

  if((prop_burn_in)<0 | (prop_burn_in)>=1 ){
    stop("prop_burn_in must be between 0 and 1.")
  }

  if((num_accept)<0 | (num_accept)>n_it*prop_burn_in ){
    stop("num_accept must be between 0 and n_it x prop_burn_in.")
  }


  ################################################################################################
  # Calculate Default Values if none supplied
  ################################################################################################

  # Offsets
  if(is.null(log_offset)==T) {
    log_offset = log(apply(counts, 2, function(x) quantile(x, probs=0.75)/median(apply(counts,2,quantile, p = 0.75))))
  }

  # Gene Names
  if(is.null(gene_names)){
    if(is.null(rownames(counts))==T){rownames(counts)<-seq(1,nrow(counts),1)}
    gene_names = rownames(counts)}

  # Prior means for log alpha
  if(is.null(prior_mean_log_alpha)){
    prior_mean_log_alpha = rep(-log(2), length(gene_names))}

  # Contrast Names
  if(run_contrasts==TRUE & is.null(contrast_names)==T){
    contrast_names = rownames(contrast_mat)}


  ############################################################################################################
  # Begin Analysis
  ############################################################################################################
  # Make sure counts are a matrix
  counts <- as.matrix(counts)

  # Create fixed effects design matrix
  design_mat <- model.matrix(fixed_effects, data=sample_data)
  fixed_effect_names <- colnames(design_mat) # save these for Brian to use for labels

  # Create a dummy contrast matrix to pass to functions if no contrasts to be run
  if(run_contrasts==F){contrast_mat = matrix(0, ncol=length(fixed_effect_names), nrow=length(fixed_effect_names))}

  # Get fixed effect starting values
  # Check that starting values have the right number of rows
  if(is.null(starting_betas)==F){
    if(nrow(starting_betas)!=nrow(counts)){
      print("The counts matrix and starting_betas indicate differing numbers of genes.  Default starting values will be used.")
      starting_betas <- matrix(0, nrow=nrow(counts), ncol=ncol(design_mat))
      starting_betas[,1] = log(rowMeans(counts))
    }
    # Check that starting values have the right number of columns
    if(ncol(starting_betas)!=ncol(design_mat)){
      print("The design matrix and starting_betas indicate differing numbers of fixed effects  Default starting values will be used.")
      starting_betas <- matrix(0, nrow=nrow(counts), ncol=ncol(design_mat))
      starting_betas[,1] = log(rowMeans(counts))
    }}


  # Set default starting values if needed
  if(is.null(starting_betas)){ starting_betas <- matrix(0, nrow=nrow(counts), ncol=ncol(design_mat))
  starting_betas[,1] = log(rowMeans(counts))}

  # If there is a contrast matrix, check that it has the right dimensions
  if(run_contrasts==TRUE){ if(ncol(contrast_mat) != ncol(design_mat)){
    stop("Design matrix and contrast matrix have differing numbers of fixed effects.")
  }}

  # Ensure that contrast, gene and fixed effect names are supplied as characters
  gene_names <- as.character(gene_names)
  contrast_names <- as.character(contrast_names)
  fixed_effect_names <- as.character(fixed_effect_names)

  # Check for random effects
  if (!is.null(random_intercept)){
    # check random_intercept %in% colnames(sample_data)
    if((random_intercept %in% colnames(sample_data))==F){
      stop('The random_intercept clustering variable cannot be found in sample_data.')
    }
    # Make the random effects design matrix
    random_int_variable <- factor(sample_data[,random_intercept], levels = unique(sample_data[,random_intercept]))
    design_mat_re <- model.matrix(~random_int_variable-1)

    if(proposal_method == 'rw'){
      ret <- nbglmm_mcmc_rw(counts = counts,
                            design_mat = design_mat,
                            design_mat_re = design_mat_re,
                            contrast_mat = contrast_mat,
                            prior_sd_betas = prior_sd_betas,
                            rw_sd_betas = rw_sd_betas,
                            prior_sd_betas_a = prior_u_re_var,
                            prior_sd_betas_b = prior_v_re_var,
                            prior_sd_rs = prior_sd_log_alpha,
                            prior_mean_log_rs = prior_mean_log_alpha,
                            n_it = n_it,
                            rw_sd_rs = rw_sd_log_alpha,
                            log_offset = log_offset,
                            starting_betas = starting_betas,
                            grain_size = 1L ,
                            prop_burn = prop_burn_in,
                            return_cont = run_contrasts,
                            # gene_names = gene_names,
                            cont_names = contrast_names,
                            beta_names = fixed_effect_names
      )
    }else{ ret <- nbglmm_mcmc_wls(counts = counts,
                                  design_mat = design_mat,
                                  design_mat_re = design_mat_re,
                                  contrast_mat = contrast_mat,
                                  prior_sd_betas = prior_sd_betas,
                                  prior_sd_betas_a = prior_u_re_var,
                                  prior_sd_betas_b = prior_v_re_var,
                                  prior_sd_rs = prior_sd_log_alpha,
                                  prior_mean_log_rs = prior_mean_log_alpha,
                                  n_it = n_it,
                                  rw_sd_rs = rw_sd_log_alpha,
                                  log_offset = log_offset,
                                  starting_betas = starting_betas,
                                  prop_burn_in = prop_burn_in,
                                  grain_size = grain_size,
                                  num_accept = num_accept,
                                  return_cont = run_contrasts,
                                  # gene_names = gene_names,
                                  cont_names = contrast_names,
                                  beta_names = fixed_effect_names
    )
    }


  }else{#do GLM
    if(proposal_method == 'rw'){ ret <- nbglm_mcmc_rw(counts = counts,
                                                      design_mat = design_mat,
                                                      contrast_mat = contrast_mat,
                                                      prior_sd_betas = prior_sd_betas,
                                                      prior_sd_rs = prior_sd_log_alpha,
                                                      prior_mean_log_rs = prior_mean_log_alpha,
                                                      n_it = n_it,
                                                      rw_sd_betas = rw_sd_betas,
                                                      rw_sd_rs = rw_sd_log_alpha,
                                                      log_offset = log_offset,
                                                      starting_betas = starting_betas,
                                                      #grain_size = grain_size,
                                                      prop_burn = prop_burn_in,
                                                      return_cont = run_contrasts,
                                                      # gene_names = gene_names,
                                                      cont_names = contrast_names,
                                                      beta_names = fixed_effect_names
    )

    }else{ ret <- nbglm_mcmc_wls(counts=counts,
                                 design_mat = design_mat,
                                 contrast_mat = contrast_mat,
                                 prior_sd_betas = prior_sd_betas,
                                 prior_sd_rs = prior_sd_log_alpha,
                                 prior_mean_log_rs = prior_mean_log_alpha,
                                 n_it = n_it,
                                 rw_sd_rs = rw_sd_log_alpha,
                                 log_offset = log_offset,
                                 starting_betas = starting_betas,
                                 grain_size = grain_size,
                                 burn_in_prop = prop_burn_in,
                                 return_cont = run_contrasts,
                                 # gene_names = gene_names,
                                 cont_names = contrast_names,
                                 beta_names = fixed_effect_names
    )}

  }
  ret$gene_names = gene_names
  ret$n_it = n_it
  ret$fixed_effects = fixed_effects
  ret$random_intercept = random_intercept
  if(run_contrasts==TRUE){ret$contrast_mat = contrast_mat
  rownames(ret$contrast_mat) <- contrast_names
  }
  ret$log_offset = log_offset
  ret$proposal_method = proposal_method
  ret$prior_sd_betas = prior_sd_betas
  ret$prior_u_re_var = prior_u_re_var
  ret$prior_v_re_var = prior_v_re_var
  ret$prior_sd_log_alpha = prior_sd_log_alpha
  ret$prior_mean_log_alpha = prior_mean_log_alpha
  ret$rw_sd_log_alpha = rw_sd_log_alpha
  ret$starting_betas = starting_betas
  ret$rw_sd_betas = rw_sd_betas
  ret$prop_burn_in = prop_burn_in
  ret$num_accept = num_accept
  return(ret)
}

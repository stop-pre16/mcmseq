#' Function to Summarize MCMSeq Models
#'
#' Creates tables of summary results from MCMSeq Model Fits.
#'
#' @param mcmseqModel mcmseq object fit using mcmseq.fit
#' @param summarizeWhat character string indicating what to summarize:"coefficient" to summarize fixed effect coefficients or "contrast" to summarize contrasts
#' @param which character string or numeric indicator of which coefficient or contrast to summarize
#' @param prop.accepts.betas minimum acceptance rate for regression coefficients to be considered converged. Default is 0.1.
#' @param prop.accepts.alphas minimum acceptance rate for dispersion parameters to be considered converged.. Default is 0.1.
#' @param geweke.p minimum Geweke p-value for a paramter to be considered converged.
#' @param order_by character string or vector of character strings which variable(s) to order the results table by: "posterior_median": order by posterior median (default decreasing order); "posterior_median_abs": order by absolute value of the median (default decreasing order); "BF": order by bayes factor (default decreasing order); "exact_pval": order by exact p-value (default increasing order); "BH_adjusted_pval": Order by adjusted p-value (default increasing order)
#' @param decreasing logical (TRUE or FALSE) indicating if the results table should appear in decreasing order
#' @param filter_by optional character string which variable to use to filter: "posterior_median": filter by posterior mean (default: greater than filter_val); "posterior_median_abs": filter by absolute value of posterior mean (default: greater than filter_val); "BF": filter by bayes factor (default: greater than filter_val); "exact_pval": filter exact p-value (default:less than filter_val); "BH_adjusted_pval": filter by adjusted p-value (default: less than filtre_val);
#' @param gt logical (TRUE or FALSE) or vector of logicals indicating if values greater than filter_val(s) should be excluded from the results table.
#' @param filter_val numeric variable or vector indicating the threshold for filtering based on the variable(s) in filter_by
#' @param log2 logical (TRUE or FALSE) indicating if log2 scale should be used (instead of natural log scale)
#'
#' @return An MCMSeq Summary Table
#'
#' @examples
#'##Fit model
#'data("simdata")
#'metadata <- simdata$metadata
#'counts <- simdata$counts[1:10,]
#'f <- ~ group*time
#'contrasts <- rbind(c(0,1,0,1), # group + group:time
#'                   c(0,0,1,1))  # time + group:time
#'rownames(contrasts) <- c("Treatment_v_Control_at_FollowUp",
#'                         "FollowUp_v_Baseline_in_Treatment")
#'fit.default <- mcmseq.fit(counts=counts,
#'                          fixed_effects = f,
#'                          sample_data = metadata,
#'                          random_intercept = 'ids',
#'                          gene_names = paste0('gene_', seq(1,10,1)),
#'                          contrast_mat = contrasts,
#'                          contrast_names = NULL,
#'                          n_it = 1000,
#'                          prop_burn_in = 0.1)
#'
#'##Summarize results for the first contrast, on the log2 scale
#'c1_table <- mcmseq.summary(mcmseqModel = fit.default,
#'                           summarizeWhat="contrast",
#'                           which = "Treatment_v_Control_at_FollowUp",
#'                           log2=TRUE)
#'
#'
#'##Filter out genes with effect sizes less than 1 and BH adjusted p-values > 0.1
#'##Order results by BH adjusted p-value
#'c1_table_filtered <- mcmseq.summary(mcmseqModel = fit.default,
#'                                    summarizeWhat="contrast",
#'                                    which = "Treatment_v_Control_at_FollowUp",
#'                                    prop.accepts.betas = c(0.1,1),
#'                                    prop.accepts.alphas  = c(0.1,0.7),
#'                                    order_by = "BH_adjusted_pval",
#'                                    decreasing = FALSE,
#'                                    filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
#'                                    filter_val = c(0.1, 1),
#'                                    gt = c(FALSE, TRUE),
#'                                    log2=TRUE)
#'
#'##Do the same for the group coefficient which represnts the difference
#'##between treatment and control at baseline
#'group_table_filtered <- mcmseq.summary(mcmseqModel = fit.default,
#'                                       summarizeWhat="coefficient",
#'                                       which = "group",
#'                                       prop.accepts.betas = c(0.1,1),
#'                                       prop.accepts.alphas  = c(0.1,0.7),
#'                                       order_by = "BH_adjusted_pval",
#'                                       decreasing = FALSE,
#'                                       filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
#'                                       filter_val = c(0.1, 1),
#'                                       gt = c(FALSE, TRUE),
#'                                       log2=TRUE)
#'
#'
#'
#'
#' @export
#'

mcmseq.summary <- function(mcmseqModel, # mcmseq object fit using mcmseq.fit
                             summarizeWhat="coefficient", #character string indicating what to summarize:"coefficient" to summarize fixed effect coefficients or "contrast" to summarize contrasts
                             which = 1,  # character string or numeric indicator of which coefficient or contrast to summarize
                             prop.accepts.betas=c(0.1), #minimum acceptance rate for regression coefficients to be considered converged. Default is 0.1.
                             prop.accepts.alphas = c(0.1), # minimum acceptance rate for dispersions to be considered converged. Default is 0.1.
                             geweke.p = 0.05,#minimum Geweke p-value for a paramter to be considered converged.
                             order_by=NULL, # character string indicating which variable to order the results table by:
                             #       "posterior_median": order by posterior median
                             #                           (default decreasing order);
                             #       "posterior_median_abs": order by absolute value of the median
                             #                               (default decreasing order);
                             #       "BF": order by bayes factor (default decreasing order);
                             #       "exact_pval": order by exact p-value (default increasing order);
                             #       "BH_adjusted_pval": Order by adjusted p-value
                             #                           (default increasing order)
                             decreasing=NULL, # logical (TRUE or FALSE) indicating if the results table should appear in decreasing order
                             filter_by=NULL, # character string or vector of character strings which variable(s) to use to filter
                             #       "posterior_median": filter by posterior mean
                             #                           (default: greater than filter_val)
                             #       "posterior_median_abs": filter by absolute value of posterior mean
                             #                               (default: greater than filter_val)
                             #       "BF": filter by bayes factor (default: greater than filter_val)
                             #       "exact_pval": filter exact p-value (default:less than filter_val)
                             #       "BH_adjusted_pval": filter by adjusted p-value
                             #                           (default: less than filtre_val)
                             gt=NULL, # logical (TRUE or FALSE) or vector of logicals indicating if values greater than filter_val(s) should be excluded from the results table
                             filter_val=NULL, # numeric variable or vector indicating the threshold which variable(s) in filter_by are filtered
                             log2=FALSE #logical (TRUE or FALSE) indicating if log2 scale should be used (instead of natural log scale)
){
  # Get Gene Names
  geneNames<-mcmseqModel$gene_names

  # Use the convergence check function to remove genes that failed to converge
  failed <- mcmseq.convergence(mcmseqModel,
                              prop.accepts.betas=prop.accepts.betas,
                              prop.accepts.alphas = prop.accepts.alphas,
                              geweke.p = geweke.p
                              )


  #Pull coefficients or contrasts based on summarizeWhat argument (Error if argument doesn't match)
  if(summarizeWhat=="contrast"){
    data_matrix<-mcmseqModel$contrast_est[which,,]
  } else if (summarizeWhat=="coefficient"){
    data_matrix<-mcmseqModel$betas_est[which,,]
  }else {
    stop('invalid summarizeWhat argument')
  }

  #Get index numbers for genes with big enough number of accepts
  #Filter out genes without big enough # of accepts from gene names and data array
  if(nrow(failed)>1){accepts_filtered<- seq(1, length(geneNames),1)[-(failed$index)]
  }else{accepts_filtered <- seq(1, length(geneNames),1)}

  geneNames_filtered<-geneNames[accepts_filtered]
  data_matrix_filtered<-data_matrix[,accepts_filtered]


  #Compute BH from exact p value
  element_BH<-p.adjust(data_matrix_filtered["p_val_exact",], method = "BH")

  #Make dataframe with estimate, SE, exact pvalue, and BH adjusted p value
  element_full_results<-data.frame(Gene=geneNames_filtered,
                                   posterior_median=data_matrix_filtered["median",],
                                   posterior_SD=data_matrix_filtered["std_dev",],
                                   BF=data_matrix_filtered["BF_exact",],
                                   exact_pval=data_matrix_filtered["p_val_exact",],
                                   BH_adjusted_pval=element_BH,
                                   posterior_median_abs=abs(data_matrix_filtered["median",]))

  #If log2=true change from LN scale to log2 scale
  if(log2==TRUE){
    element_full_results$posterior_median=log2(exp(element_full_results$posterior_median))
    element_full_results$posterior_median_abs=log2(exp(element_full_results$posterior_median_abs))
    element_full_results$posterior_SD=log2(exp(element_full_results$posterior_SD))
  } else if(!is.logical(log2)){
    stop("log2 must be logical")
  }

  #If there is a filter value, needs to be a filter by value
  if(!is.null(filter_by)&is.null(filter_val)){
    stop('Must supply a value for filter_val')
  }

  #Values of gt, filter_val, and filter_by need to be the same length
  if(length(filter_by)!=length(filter_val)| length(filter_by)!=length(gt) & !is.null(gt)){
    stop('filter_by, filter_val, and gt vectors must be the same length')
  }

  #Filter value must be numeric
  if(!is.numeric(filter_val) & !is.null(filter_val)){
    stop("filter_val must be numeric")
  }

  #Filter data by specified column and value
  #Stop if invalid filter_by argument
  ###Option to choose which direction to filter (greater than or less than)?

  element_sig_results<-element_full_results
  if(length(filter_by)>0){
  for( x in 1:length(filter_by)){
    if(!(filter_by[x] %in% c(colnames(element_sig_results)))){
      stop('invalid filter_by argument')
    }else if(is.null(gt)){
      if(filter_by[x]=="exact_pval" | filter_by[x]=="BH_adjusted_pval"){
        element_sig_results<-element_sig_results[
          element_sig_results[,filter_by[x]]<filter_val[x],]
      }else{
        element_sig_results<-element_sig_results[
          element_sig_results[,filter_by[x]]>filter_val[x],]
      }}else{
        if(!is.logical(gt[x])){
          stop("gt must be logical")
        }else if(gt[x]==TRUE){
          element_sig_results<-element_sig_results[
            element_sig_results[,filter_by[x]]>filter_val[x],]
        } else{
          element_sig_results<-element_sig_results[
            element_sig_results[,filter_by[x]]<filter_val[x],]
        }
      }
  }}




  #Order by column specified in order_by argument
  #Option to choose order
  if (is.null(order_by)){
    element_sig_results=element_sig_results
  }else if(!(order_by %in% colnames(element_full_results))){
    stop('invalid order_by argument')

  }else if(is.null(decreasing)){
    if(order_by=="exact_pval" | order_by=="BH_adjusted_pval") {
      element_sig_results<-element_sig_results[order(
        element_sig_results[, order_by], decreasing = FALSE),]
    } else{
      element_sig_results<-element_sig_results[order(
        element_sig_results[,order_by], decreasing = TRUE),]
    }
  }else{
    if(!is.logical(decreasing)){
      stop('decreasing must be logical')
    }else{
      element_sig_results<-element_sig_results[order(
        element_sig_results[, order_by], decreasing = decreasing),]
    }}

  #Return table
  element_sig_results<-element_sig_results[,-length(element_sig_results)]
  return(element_sig_results)

}


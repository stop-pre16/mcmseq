#' Function to Calculate DESeq2 Size Factors
#'
#' Estimates the size factors from DESeq2 to be used as offsets in NBGLMM model fits
#'
#' @param counts A (G x N) numeric matrix or data frame of RNA-seq counts, with genes in rows and samples in columns. G = number of genes.  N = number of samples.
#'
#'
#' @return A vector with the size factors for each of the N samples
#'
#' @examples
#'data("simdata")
#'counts <- simdata$counts
#'
#'##Calculate size factors
#'size.factors <- est_DESeq2_size_factors(counts)
#'
#'
#'
#' @export
#'
#'

est_DESeq2_size_factors = function(counts){
  geo_means = apply(X = counts, MARGIN = 1, FUN = function(x){
    if (all(x == 0)){
      ret = 0
    }
    else {
      ret = exp(sum(log(x[x > 0])) / length(x[x > 0]))
    }
    return(ret)
  })

  est_sf = apply(X = counts, MARGIN = 2, FUN = function(x){
    ratio_vec = x / geo_means
    ret = median(ratio_vec[ratio_vec > 0])
    return(ret)
  })
  return(est_sf)
}

#' Function to Create Trended Priors for the Dispersion Parameter
#'
#' Fits loess regression to naive estimates of log(dispersion) vs. log(mean counts) to develop gene-specific priors for the dispersion parameters.
#'
#' @param counts A (G x N) numeric matrix or data frame of RNA-seq counts, with genes in rows and samples in columns. G = number of genes.  N = number of samples.
#' @param span An optional numeric paramter that controls the degree of smoothing.  Defaults to 1.  Higher values result in a smoother curve.
#' #'
#' @return A list including prior_mean_log_alpha and prior_sd_log_alpha which can be used in the mcmseq.fit function. A plot showing the loess fit is also generated.
#'
#' @examples
#'
#'
#' @export
#'

trended.dispersion <- function(counts, span=1){# First calculate counts per million for each sample
  cpms <- 1000000*apply(counts, 2, function(x) x/sum(x))

  # Normalize the counts to the median library size
  normalized_counts <- cpms*median(colSums(counts))/1000000

  # Calculate the log(mean) and method of moments dispersion estimator for each gene
  # log means
  log_means <- log(rowMeans(normalized_counts))

  # variance
  vars <- apply(normalized_counts, 1, var)
  mom_dispersion <- (vars - exp(log_means))/(exp(log_means)^2)

  # Fit the Loess model
  m1 <- loess(log(mom_dispersion) ~ log_means, span=span)

  # Get predicted values for log(dispersion)
  m1.predictions <- predict(m1)

  # Save the SD of the residuals
  sd_res <- sd(m1$residuals)

  # Plot the predicted values.  If the line is too wiggly, increase the span argument
  # in the loess function above.
  plot(log_means, log(mom_dispersion), type="p", ylab = "ln(MOM Dispersion Estimate)",
       xlab = "ln(Mean Counts)")
  lines(m1.predictions[order(log_means)], x=log_means[order(log_means)], col="red")

  return (list(prior_mean_log_alpha = m1.predictions, prior_sd_log_alpha = sd_res))
  }

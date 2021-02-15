## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
# Load the library
library(mcmseq)

# Load the data
data("simdata")

names(simdata)

# metadata has information about the the study design
# each row is a sample and corresponds to a column
# in the counts matrix
metadata <- simdata$metadata
metadata

# The counts matrix has the same number of columns as
# rows in metadata.  The columns of the counts matrix
# are in the same order as the rows of the metadata.
counts <- simdata$counts
head(counts)


## ------------------------------------------------------------------------
# Create the fixed effects model formula
f <- ~ group*time


## ------------------------------------------------------------------------
# Look at the fixed effects design matrix to develop the contrast_mat
head(model.matrix(f, data = metadata))

# Create the contrast matrix
contrasts <- rbind(c(0,1,0,1), # group + group:time
                   c(0,0,1,1)  # time + group:time 
              )

# Name the contrasts by specifying rownames
rownames(contrasts) <- c("Treatment_v_Control_at_FollowUp", 
                         "FollowUp_v_Baseline_in_Treatment")

# View the contrast matrix
contrasts

## ------------------------------------------------------------------------
# Fit the Model
fit.default <- mcmseq.fit(counts=counts, 
                       fixed_effects = f,
                       sample_data = metadata,
                       random_intercept = 'ids', 
                       gene_names = paste0('gene_', seq(1,1000,1)),
                       contrast_mat = contrasts,
                       contrast_names = NULL,
                       n_it = 1000,
                       prop_burn_in = 0.1
)

# Look at the mcmseq fit object
class(fit.default)

names(fit.default)

## ------------------------------------------------------------------------
# See regression coefficient estimates for the first gene
fit.default$betas_est[,,1]

# See the contrast estimates for the first gene
fit.default$contrast_est[,,1]

# See the dispersion and random intercept variance estimates for the
# first 10 genes
fit.default$alphas_est[1:10]
fit.default$sig2_est[1:10]


## ------------------------------------------------------------------------
# See convergence failures, the Geweke p-values and acceptance rates
fit.default$convergence_failures

## ------------------------------------------------------------------------
# Summarize results for the first contrast, on the log2 scale
c1_table <- mcmseq.summary(mcmseqModel = fit.default, 
                 summarizeWhat="contrast",  
                 which = "Treatment_v_Control_at_FollowUp",
                  log2=TRUE 
            )

dim(c1_table)

head(c1_table)

# Filter out genes with effect sizes less than 1 and BH adjusted p-values > 0.05
# Order results by BH adjusted p-value
c1_table_filtered <- mcmseq.summary(mcmseqModel = fit.default, 
                 summarizeWhat="contrast",  
                 which = "Treatment_v_Control_at_FollowUp",  
                 prop.accepts.betas = c(0.1,1), 
                 prop.accepts.alphas  = c(0.1,0.7),
                 order_by = "BH_adjusted_pval", 
                 decreasing = FALSE,
                 filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
                 filter_val = c(0.05, 1),
                 gt = c(FALSE, TRUE),
                  log2=TRUE 
            )

dim(c1_table_filtered)

head(c1_table_filtered)

# Do the same for the group coefficient which represnts the difference 
# between treatment and control at baseline
group_table_filtered <- mcmseq.summary(mcmseqModel = fit.default, 
                 summarizeWhat="coefficient",  
                 which = "group",  
                 prop.accepts.betas = c(0.1,1), 
                 prop.accepts.alphas  = c(0.1,0.7),
                 order_by = "BH_adjusted_pval", 
                 decreasing = FALSE,
                 filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
                 filter_val = c(0.05, 1),
                 gt = c(FALSE, TRUE),
                  log2=TRUE 
            )

dim(c1_table_filtered)

head(c1_table_filtered)


## ------------------------------------------------------------------------
# Develop the trended dispersion prior
disp.prior <- trended.dispersion(counts, span=1)


## ------------------------------------------------------------------------
# Fit the Model
fit.custom <- mcmseq.fit(counts=counts, 
                       fixed_effects = f,
                       sample_data = metadata,
                       random_intercept = 'ids', 
                       gene_names = paste0('gene_', seq(1,1000,1)),
                       contrast_mat = contrasts,
                       contrast_names = NULL,
                       prior_mean_log_alpha = disp.prior$prior_mean_log_alpha,
                       prior_sd_log_alpha = 2 * disp.prior$prior_sd_log_alpha,
                       n_it = 1000,
                       prop_burn_in = 0.1
)


## ------------------------------------------------------------------------
# Summarize results for the first contrast, on the log2 scale
c1_table_custom <- mcmseq.summary(mcmseqModel = fit.custom, 
                 summarizeWhat="contrast",  
                 which = "Treatment_v_Control_at_FollowUp",
                  log2=TRUE 
            )

dim(c1_table_custom)

head(c1_table_custom)

# Filter out genes with effect sizes less than 1 and BH adjusted p-values > 0.05
# Order results by BH adjusted p-value
c1_table_custom_filtered <- mcmseq.summary(mcmseqModel = fit.custom, 
                 summarizeWhat="contrast",  
                 which = "Treatment_v_Control_at_FollowUp",  
                 prop.accepts.betas = c(0.1), 
                 prop.accepts.alphas  = c(0.1),
                 order_by = "BH_adjusted_pval", 
                 decreasing = FALSE,
                 filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
                 filter_val = c(0.05, 1),
                 gt = c(FALSE, TRUE),
                  log2=TRUE 
            )

dim(c1_table_custom_filtered)

head(c1_table_custom_filtered)



## ------------------------------------------------------------------------
# Load the data
data("simdata")

names(simdata)

# metadata has information about the the study design
# each row is a sample and corresponds to a column
# in the counts matrix
metadata <- simdata$metadata

# Subset the metadata to the follow-up observations
metadata.glm <- metadata[metadata$time==1,]

# The counts matrix has the same number of columns as
# rows in metadata.  The columns of the counts matrix
# are in the same order as the rows of the metadata.
counts <- simdata$counts
head(counts)

# Subset the metadata to the follow-up observations
counts.glm <- counts[,metadata$time==1]


## ------------------------------------------------------------------------
# Fit the Model
fit.default.glm <- mcmseq.fit(counts=counts.glm, 
                       fixed_effects = ~ group,
                       sample_data = metadata.glm,
                       random_intercept = NULL, 
                       gene_names = paste0('gene_', seq(1,1000,1)),
                       contrast_mat = NULL,
                       contrast_names = NULL,
                       n_it = 1000,
                       prop_burn_in = 0.1
)

## ------------------------------------------------------------------------
group_table_filtered <- mcmseq.summary(mcmseqModel = fit.default, 
                 summarizeWhat="coefficient",  
                 which = "group",  
                 prop.accepts.betas = c(0.1,1), 
                 prop.accepts.alphas  = c(0.1,0.7),
                 order_by = "BH_adjusted_pval", 
                 decreasing = FALSE,
                 filter_by = c("BH_adjusted_pval", "posterior_median_abs"),
                 filter_val = c(0.05, 1),
                 gt = c(FALSE, TRUE),
                  log2=TRUE 
            )

dim(c1_table_filtered)

head(c1_table_filtered)


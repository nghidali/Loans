library(tidyverse)
library(tidymodels)
set.seed(42)

load("mars_tuned_inputs.rda")
control <- control_resamples(verbose = TRUE)
mars_tuned <- mars_workflow %>%
  tune_grid(loan_folds, grid = mars_grid, control = control)
saveRDS(mars_tuned, "mars_tuned.rds")

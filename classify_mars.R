# Load libraries and set seed
library(tidyverse)
library(tidymodels)
library(lubridate)
set.seed(42)

# process training set
loans <- read_csv("data/train.csv")  %>%
  select(-purpose) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = as.numeric(parse_date(earliest_cr_line, format = "%b-%Y")),
    emp_length = factor(
      emp_length,
      ordered = TRUE,
      levels = c(
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
        "n/a"
      )
    ),
    emp_title = factor(emp_title),
    grade = factor(grade, ordered = TRUE),
    home_ownership = factor(home_ownership),
    initial_list_status = factor(initial_list_status),
    last_credit_pull_d = as.numeric(parse_date(last_credit_pull_d, format = "%b-%Y")),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status),
    hi_int_prncp_pd = factor(hi_int_prncp_pd)
  )

# process testing set
testing_data <- read_csv("data/test.csv")  %>%
  select(-purpose) %>%
  mutate(
    addr_state = factor(addr_state),
    application_type = factor(application_type),
    earliest_cr_line = as.numeric(parse_date(earliest_cr_line, format = "%b-%Y")),
    emp_length = factor(
      emp_length,
      ordered = TRUE,
      levels = c(
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
        "n/a"
      )
    ),
    emp_title = factor(emp_title),
    grade = factor(grade, ordered = TRUE),
    home_ownership = factor(home_ownership),
    initial_list_status = factor(initial_list_status),
    last_credit_pull_d = as.numeric(parse_date(last_credit_pull_d, format = "%b-%Y")),
    sub_grade = factor(sub_grade, ordered = TRUE),
    term = factor(term),
    verification_status = factor(verification_status)
  )

# no split needed since test set is already seperate
loan_folds <- vfold_cv(data = loans, v = 10, repeats = 3, strata = hi_int_prncp_pd)
ggplot(loans) +
  geom_bar(mapping = aes(hi_int_prncp_pd))

# Create recipe, remove id column
loan_recipe1 <- recipe(hi_int_prncp_pd ~ ., data = loans) %>%
  step_rm(contains("id")) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Make model
mars_model <- mars(mode = "classification",
                   num_terms = tune(),
                   prod_degree = tune()) %>%
  set_engine("earth")

### Make workflow
mars_workflow <- workflow() %>%
  add_recipe(loan_recipe1) %>%
  add_model(mars_model)

# Update tuning parameters
bt_params <- parameters(bt_workflow) %>%
  update(mtry = mtry(range = c(1,18)),
         learn_rate = learn_rate(range = c(-5, -1)))
bt_grid <- grid_regular(bt_params,levels = 3)

mars_params <- parameters(mars_workflow) %>%
  update(num_terms = num_terms(range = c(1,10)))
mars_grid <- grid_regular(mars_params, levels = 3)

# Save output
save(mars_workflow, loan_folds, mars_grid, file = "mars_tuned_inputs.rda")

# --- Run bt_tune.R ---

# Load tuned boosted tree
mars_tuned <- readRDS("mars_tuned.rds")

# Pick optimal tuning params
show_best(mars_tuned, metric = "accuracy")
mars_results <- mars_workflow %>%
  finalize_workflow(select_best(mars_tuned, metric = "accuracy")) %>%
  fit(loans)

# Predict test set
mars_predictions <- predict(mars_results, new_data = testing_data) %>%
  bind_cols(testing_data %>% select(id)) %>%
  rename(
    Category = .pred_class,
    Id = id
  )

# Write out predictions
write_csv(mars_predictions, "mars_predictions.csv")



# Use-KNearestNeighbourRegressor-to-Predict-Price-of-House.

sacr_recipe <- recipe(price ~ sqft, data = sacramento_train) |>
  step_scale(all_predictors()) |>
  step_center(all_predictors())

sacr_spec <- nearest_neighbor(weight_func = "rectangular", 
                              neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("regression")

sacr_vfold <- vfold_cv(sacramento_train, v = 5, strata = price)

sacr_wkflw <- workflow() |>
  add_recipe(sacr_recipe) |>
  add_model(sacr_spec)

sacr_wkflw
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: nearest_neighbor()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 2 Recipe Steps
## 
## • step_scale()
## • step_center()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## K-Nearest Neighbor Model Specification (regression)
## 
## Main Arguments:
##   neighbors = tune()
##   weight_func = rectangular
## 
## Computational engine: kknn

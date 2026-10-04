# Methodology

## Workflow

1. Read the feature and target CSV files.
2. Merge them using `jobId`.
3. Remove salary values less than or equal to zero.
4. Remove `jobId` from the predictor set.
5. Split the data into 80% training and 20% held-out test data using `random_state=42`.
6. One-hot encode categorical variables inside a scikit-learn `ColumnTransformer`.
7. Fit exactly three regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
8. Evaluate each model with:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R²

## Parameters used

### Decision Tree

- `max_depth=12`
- `min_samples_leaf=10`
- `random_state=42`

### Random Forest

- `n_estimators=80`
- `max_depth=16`
- `min_samples_leaf=3`
- `max_features=0.8`
- `random_state=42`
- `n_jobs=-1`

## Important interpretation

The notebook's saved run uses the first 100,000 rows of the training data for practical
runtime. One invalid zero-salary row was removed, leaving 99,999 observations.

Linear Regression and Random Forest have nearly identical held-out R² values. The
difference is only about 0.000009, so the project should describe their predictive
performance as practically tied rather than claiming a meaningful advantage.

Linear Regression is nevertheless a strong practical choice in the saved run because
it achieved the marginally highest R² while training far faster than Random Forest.

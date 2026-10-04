# Data Dictionary

The modelling table contains 9 columns after merging the feature and salary files.

| Variable | Role | Type | Notes |
|---|---|---|---|
| `jobId` | Identifier | Categorical/ID | Removed before modelling |
| `companyId` | Predictor | Categorical | Encoded using one-hot encoding |
| `jobType` | Predictor | Categorical | Examples include CEO, CFO, Manager, Junior |
| `degree` | Predictor | Categorical | Education level |
| `major` | Predictor | Categorical | Academic major |
| `industry` | Predictor | Categorical | Industry group |
| `yearsExperience` | Predictor | Numeric | Total professional experience |
| `milesFromMetropolis` | Predictor | Numeric | Distance from a metropolis |
| `salary` | Target | Numeric | Salary to be predicted |

## Model input

The final models therefore use **7 predictors**:

1. `companyId`
2. `jobType`
3. `degree`
4. `major`
5. `industry`
6. `yearsExperience`
7. `milesFromMetropolis`

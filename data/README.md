# Dataset

The raw dataset is **not included** in this repository.

## Source

Dataset reference:

- **Salary Prediction** by Jenny Chou
  https://github.com/jenny-chou/Salary-Prediction
- The source repository provides `SalaryPredictions.zip`.

The ZIP contains:

```text
data/
├── train_features.csv
├── train_salaries.csv
└── test_features.csv
```

For this project, download `SalaryPredictions.zip` and place it here:

```text
data/SalaryPredictions.zip
```

Both the notebook and `src/train_models.py` can extract it automatically.

## Training schema

`train_features.csv` contains 8 columns:

| Column | Meaning |
|---|---|
| `jobId` | Unique job identifier |
| `companyId` | Company identifier |
| `jobType` | Job level/type |
| `degree` | Highest degree category |
| `major` | Academic major |
| `industry` | Industry category |
| `yearsExperience` | Total years of work experience |
| `milesFromMetropolis` | Distance from a metropolitan area |

`train_salaries.csv` contains:

| Column | Meaning |
|---|---|
| `jobId` | Job identifier used for merging |
| `salary` | Target variable |

After merging, `jobId` is dropped from the model because it is an identifier.

> `yearsExperience` means total professional experience. It does **not** represent
> the number of years the person has worked at the current company.

## Redistribution note

The referenced source repository does not declare a GitHub license. For that reason,
this portfolio repository does not redistribute the raw dataset. Review the source
terms before publishing or redistributing the data yourself.

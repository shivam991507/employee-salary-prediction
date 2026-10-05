"""
Employee Salary Prediction
Models: Linear Regression, Decision Tree Regressor, Random Forest Regressor

The script mirrors the modelling workflow used in the notebook.
"""

from pathlib import Path
import argparse
import time
import zipfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42


def locate_or_extract_data(data_dir: Path):
    """Return paths to train_features.csv and train_salaries.csv."""
    feature_path = data_dir / "train_features.csv"
    salary_path = data_dir / "train_salaries.csv"

    if feature_path.exists() and salary_path.exists():
        return feature_path, salary_path

    zip_path = data_dir / "SalaryPredictions.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find CSV files or {zip_path}. "
            "See data/README.md for download instructions."
        )

    extract_dir = data_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    candidates = [
        (extract_dir / "data" / "train_features.csv",
         extract_dir / "data" / "train_salaries.csv"),
        (extract_dir / "train_features.csv",
         extract_dir / "train_salaries.csv"),
    ]
    for f_path, s_path in candidates:
        if f_path.exists() and s_path.exists():
            return f_path, s_path

    raise FileNotFoundError("Training CSV files were not found after extraction.")


def load_data(data_dir: Path, nrows: int | None):
    feature_path, salary_path = locate_or_extract_data(data_dir)

    features = pd.read_csv(feature_path, nrows=nrows)
    salaries = pd.read_csv(salary_path, nrows=nrows)

    data = features.merge(salaries, on="jobId", how="inner")

    # Salary <= 0 is invalid for this regression problem.
    data = data[data["salary"] > 0].copy()
    return data


def build_preprocessor():
    categorical_features = [
        "companyId",
        "jobType",
        "degree",
        "major",
        "industry",
    ]
    numeric_features = [
        "yearsExperience",
        "milesFromMetropolis",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_models(data: pd.DataFrame):
    X = data.drop(columns=["salary", "jobId"])
    y = data["salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = []

    for name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )

        start = time.time()
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        elapsed = time.time() - start

        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
                "Training_seconds": elapsed,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing SalaryPredictions.zip or extracted training CSVs.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=100_000,
        help="Number of training rows to read. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/model_comparison_reproduced.csv"),
    )
    args = parser.parse_args()

    nrows = None if args.nrows == 0 else args.nrows
    data = load_data(args.data_dir, nrows=nrows)

    print(f"Rows used after cleaning: {len(data):,}")
    results = evaluate_models(data)
    print(results.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "polars",
#     "scikit-learn",
#     "embetter",
#     "sentence-transformers",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from embetter.text import SentenceEncoder
    import sentence_transformers
    return (
        CountVectorizer,
        DummyClassifier,
        HistGradientBoostingClassifier,
        KNeighborsClassifier,
        LogisticRegression,
        Pipeline,
        SentenceEncoder,
        StandardScaler,
        StratifiedKFold,
        cross_validate,
        mo,
        np,
        pl,
    )


@app.cell
def _(mo, pl):
    # Load the Titanic dataset
    df = pl.read_csv("titanic.csv")
    mo.md(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return (df,)


@app.cell
def _(df, mo, pl):
    # Check for missing values
    missing_values = pl.DataFrame({
        "Column": df.columns,
        "Missing Count": [df[col].null_count() for col in df.columns],
        "Missing %": [(df[col].null_count() / len(df) * 100) for col in df.columns]
    })

    missing_values = missing_values.filter(pl.col("Missing Count") > 0)

    mo.md("## Missing Values Summary")
    missing_values
    return


@app.cell
def _(df, mo, pl):
    # Preprocess the data
    df_processed = (
        df
        # Fill missing values
        .with_columns([
            pl.col("age").fill_null(pl.col("age").median()),
            pl.col("fare").fill_null(pl.col("fare").mean()),
        ])
        # Create binary encoding for sex
        .with_columns([
            (pl.col("sex") == "male").cast(pl.Int64).alias("is_male")
        ])
        # Select relevant features
        .select([
            "survived",
            "pclass",
            "is_male", 
            "age",
            "fare",
            "sibsp",
            "parch"
        ])
    )

    # Convert to numpy arrays for sklearn
    X = df_processed.drop("survived").to_numpy()
    y = df_processed["survived"].to_numpy()

    # Also extract names for text-based model
    X_names = df["name"].to_numpy()

    mo.md(f"Prepared dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, X_names, y


@app.cell
def _(
    CountVectorizer,
    DummyClassifier,
    HistGradientBoostingClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    Pipeline,
    SentenceEncoder,
    StandardScaler,
    StratifiedKFold,
    X,
    X_names,
    cross_validate,
    mo,
    np,
    pl,
    y,
):
    # Define models using numerical features
    models_numerical = {
        "Dummy (Most Frequent)": (DummyClassifier(strategy="most_frequent"), X),
        "Logistic Regression": (Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]), X),
        "K-Nearest Neighbors": (Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]), X),
        "Histogram Gradient Boosting": (HistGradientBoostingClassifier(random_state=42), X)
    }

    # Add text-based model using CountVectorizer
    text_model = Pipeline([
        ("vectorizer", CountVectorizer(
            token_pattern=r'\b\w+\b',  # Extract all words
            lowercase=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    models_numerical["Names (CountVectorizer + LogReg)"] = (text_model, X_names)

    # Add embedding-based model using SentenceEncoder
    embedding_model = Pipeline([
        ("embedder", SentenceEncoder('all-MiniLM-L6-v2')),  # Lightweight but effective model
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    models_numerical["Names (Embeddings + LogReg)"] = (embedding_model, X_names)

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    # Train and evaluate all models
    results = {}
    for name, (model, features) in models_numerical.items():
        cv_scores = cross_validate(
            model, features, y, 
            cv=cv, 
            scoring=scoring,
            return_train_score=False
        )
        results[name] = cv_scores

    # Create results table with mean ± std
    def format_metric(scores):
        mean = np.mean(scores)
        std = np.std(scores)
        return f"{mean:.3f} ± {std:.3f}"

    results_data = []
    for model_name, scores in results.items():
        results_data.append({
            "Model": model_name,
            "Accuracy": format_metric(scores['test_accuracy']),
            "Precision": format_metric(scores['test_precision']),
            "Recall": format_metric(scores['test_recall']),
            "F1-Score": format_metric(scores['test_f1'])
        })

    results_df = pl.DataFrame(results_data)

    # Sort by accuracy (extract mean value for sorting)
    results_df = results_df.sort(
        by=pl.col("Accuracy").str.extract(r"(\d+\.\d+)").cast(pl.Float64),
        descending=True
    )

    mo.md("## Model Comparison Results (5-Fold Cross-Validation)")
    results_df
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preparing a churn dataset

    Nothing too fancy, just some features. 
    """
    )
    return


@app.cell
def _(mo):
    df_pl = mo.sql(
        f"""
        select * from read_parquet("wow-full.parquet")
        """
    )
    return (df_pl,)


@app.cell
def _(df_pl):
    import polars as pl
    import datetime as dt

    # Calculate the minimum and maximum dates in the dataset
    min_date = df_pl["datetime"].min()
    max_date = df_pl["datetime"].max()

    # Calculate the halfway point as our strike date
    strike_date = min_date + (max_date - min_date) / 2
    return dt, pl, strike_date


@app.cell
def _(df_pl, dt, pl, strike_date):
    # Create a function that will determine if a user churned
    def calculate_churn_features(df, strike_date):
        # Convert strike_date to datetime if needed
        if isinstance(strike_date, str):
            strike_date = dt.datetime.fromisoformat(strike_date)
    
        # End of observation period (2 months after strike date)
        end_date = strike_date + dt.timedelta(days=60)
    
        # Pre-strike data (only use data before the strike date)
        pre_strike_df = df.filter(pl.col("datetime") < strike_date)
    
        # Post-strike data (for determining churn)
        post_strike_df = df.filter(
            (pl.col("datetime") >= strike_date) & 
            (pl.col("datetime") <= end_date)
        )
    
        # Calculate features from pre-strike data
        features = (
            pre_strike_df
            .group_by("player_id")
            .agg([
                pl.col("class").first(),
                pl.col("race").first(),
                pl.col("level").max().alias("max_level"),
                pl.col("level").min().alias("min_level"),
                pl.col("level").mean().alias("avg_level"),
                pl.col("datetime").n_unique().alias("days_active_pre_strike"),
                pl.col("datetime").max().alias("last_activity_date"),
                pl.col("where").n_unique().alias("num_locations_visited")
            ])
        )
    
        # Calculate days since last activity
        features = features.with_columns(
            (strike_date - pl.col("last_activity_date"))
            .dt.total_days()
            .alias("days_since_last_activity")
        )
    
        # Determine churn (1 if no activity post-strike, 0 if activity observed)
        active_players_post = post_strike_df.select("player_id").unique()
    
        # Add churn indicator
        all_players = features.select("player_id")
        churned_players = all_players.join(
            active_players_post, 
            on="player_id", 
            how="anti"
        )
    
        features = features.with_columns(
            pl.lit(0).alias("churned")
        )
    
        features = features.with_columns([
            pl.when(pl.col("player_id").is_in(churned_players["player_id"]))
            .then(1)
            .otherwise(0)
            .alias("churned")
        ])
    
        return features

    # Calculate the features
    churn_features = calculate_churn_features(df_pl, strike_date)

    # Display the first few rows of the features
    churn_features
    return (churn_features,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preparing the database

    Let's do sqlite here.
    """
    )
    return


@app.cell
def _():
    import sqlalchemy

    DATABASE_URL = "sqlite:///wow.sqlite"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(churn_features, engine):
    ## write dataframe into engine
    import pandas as pd
    _df = churn_features.to_pandas()
    _df.to_sql('churn_features', engine, if_exists='replace', index=False)
    return


@app.cell
def _(churn_features):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.compose import ColumnTransformer

    X = churn_features.select("race", "class", "max_level", "min_level", "avg_level", "days_active_pre_strike", "num_locations_visited", "days_since_last_activity")
    y = churn_features.select("churned").to_numpy()[:, 0]


    pipeline = make_pipeline(
        ColumnTransformer([
            ("onehot", OneHotEncoder(), ["race", "class"]),
            ("scaler", StandardScaler(), ["max_level", "min_level", "avg_level", "days_active_pre_strike", "num_locations_visited", "days_since_last_activity"]),
        ]),
        GradientBoostingClassifier(),
    )

    pipeline.fit(X, y)
    return X, pipeline


@app.cell
def _(X, pipeline):
    pipeline.predict(X)
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## This is Orbit

    It does what it does
    """
    )
    return


@app.cell
def _(X, pipeline):
    import orbital 

    orbital_pipeline = orbital.parse_pipeline(
        pipeline, 
        features= orbital.types.guess_datatypes(X)
    )

    print(orbital_pipeline)
    return orbital, orbital_pipeline


@app.cell
def _(orbital, orbital_pipeline):
    from mohtml import p

    sql_out = orbital.export_sql(
        "churn_features", orbital_pipeline, dialect="sqlite"
    )
    p(sql_out)
    return (sql_out,)


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Let's compare

    Two ways of doing it
    """
    )
    return


@app.cell
def _(engine, mo, sql_out):
    preds_out = mo.sql(sql_out, engine=engine)
    preds_out
    return (preds_out,)


@app.cell
def _(X, pipeline):
    pipeline.predict(X)[:10]
    return


@app.cell
def _(X, pipeline, preds_out):
    import matplotlib.pylab as plt

    plt.hist(pipeline.predict_proba(X)[:, 0])
    plt.hist(preds_out["output_probability.0"].to_numpy())
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

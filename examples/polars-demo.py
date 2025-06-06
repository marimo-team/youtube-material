import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    alt.data_transformers.disable_max_rows()
    return mo, pl


@app.cell
def _(mo):
    mo.md(
        """
    ## Playing around with polars and WoW

    This notebook has a fun dataset that we will explore with polars, altair and friends. To get started though, you'll want to download [this dataset](https://github.com/koaning/wow-avatar-datasets/blob/main/wow-full.parquet) manually beforehand.
    """
    )
    return


@app.cell
def _(pl):
    df = pl.read_parquet("~/Downloads/wow-full.parquet")
    df
    return (df,)


@app.cell(hide_code=True)
def _(df, mo, pl):
    tbl = mo.ui.table(
        df
        .group_by("player_id")
        .len()
        .filter(pl.col("len") > 1000)
        .sort("len")
    )
    tbl
    return (tbl,)


@app.cell
def _(tbl):
    player_ids = [_ for _ in tbl.value["player_id"]]
    return (player_ids,)


@app.cell
def _(df, pl, player_ids):
    chart1 = (df
        .filter(pl.col("player_id").is_in(player_ids))
        .plot
        .line(x="datetime", y="level", color="player_id:N")
        .properties(title="WoW players level up over time") 
    )

    chart2 = (df
        .filter(pl.col("player_id").is_in(player_ids))
        .plot
        .scatter(x="datetime", y="level", color="player_id:N")
        .properties(title="WoW players level up over time")
    )

    chart1 + chart2
    return


@app.cell
def _(df, pl):
    df_time_taken = (
        df
        .group_by("class", "level")
        .agg(
            len=pl.len(), 
            nchar=pl.n_unique("player_id")
        )
        .filter(
            pl.col("level") != 60, 
            pl.col("level") != 70, 
            pl.col("level") != 80
        )
        .with_columns(hours_per_char=pl.col("len")/pl.col("nchar") / 6)
    )

    df_time_taken.plot.line(x="level", y="hours_per_char", color="class:N")
    return (df_time_taken,)


@app.cell
def _(df, pl):
    df_time_taken_guild = (
        df
        .with_columns(~pl.col("guild").is_null())
        .filter(pl.col("class") != "Death Knight")
        .group_by("guild", "level")
        .agg(
            len=pl.len(), 
            nchar=pl.n_unique("player_id")
        )
        .filter(
            pl.col("level") != 60, 
            pl.col("level") != 70, 
            pl.col("level") != 80
        )
        .with_columns(hours_per_char=pl.col("len")/pl.col("nchar") / 6)
    )

    df_time_taken_guild.plot.line(x="level", y="hours_per_char", color="guild:N")
    return


@app.cell
def _(df_time_taken, pl):
    (
        df_time_taken
        .filter(pl.col("level") < 60)
        .group_by("class")
        .agg(total_hours=pl.col("hours_per_char").sum())
        .sort("total_hours")
        .filter(~pl.col("class").is_in(["482", "Death Knight", "3485伊"]))
    )
    return


@app.cell
def _(mo):
    mo.md("""Let's clean the dataset up and set up a proper pipeline. After all ... maybe we have some bots in here?""")
    return


@app.cell
def _(df, pl):
    def set_types(dataf):
        return (dataf.with_columns([
                    pl.col("guild").is_not_null(),
                    pl.col("datetime").cast(pl.Int64).alias("timestamp")
                ]))

    def clean_data(dataf):
        return (
            dataf
            .filter(
                ~pl.col("class").is_in(["482", "Death Knight", "3485伊", "2400"]),
                pl.col("race").is_in(["Troll", "Orc", "Undead", "Tauren", "Blood Elf"])
            )
        )

    def sessionize(dataf, threshold=20 * 60 * 1000):
        return (dataf
                 .sort(["player_id", "timestamp"])
                 .with_columns(
                     (pl.col("timestamp").diff().cast(pl.Int64) > threshold).fill_null(True).alias("ts_diff"),
                     (pl.col("player_id").diff() != 0).fill_null(True).alias("char_diff"),
                 )
                 .with_columns(
                     (pl.col("ts_diff") | pl.col("char_diff")).alias("new_session_mark")
                 )
                 .with_columns(
                     pl.col("new_session_mark").cum_sum().alias("session")   
                 )
                 .drop(["char_diff", "ts_diff", "new_session_mark"]))

    def add_features(dataf):
        return (dataf
                 .with_columns(
                     pl.col("player_id").count().over("session").alias("session_length"),
                     pl.col("session").n_unique().over("player_id").alias("n_sessions_per_char")
                 ))

    def remove_bots(dataf, max_session_hours=24):
        # We're using some domain knowledge. The logger of our dataset should log
        # data every 10 minutes. That's what this line is based on.
        n_rows = max_session_hours * 6
        return (dataf
                .filter(pl.col("session_length").max().over("player_id") < n_rows))

    cached = (
        df
        .pipe(set_types)
        .pipe(clean_data)
        .pipe(sessionize, threshold=30 * 60 * 1000)
        .pipe(add_features)
    )
    return cached, remove_bots


@app.cell
def _(cached, max_session_threshold, remove_bots):
    df_out = (
        cached.pipe(remove_bots, max_session_hours=max_session_threshold.value)
    )
    return (df_out,)


@app.cell(hide_code=True)
def _(mo):
    session_threshold = mo.ui.slider(20, 120, 10, label="Session threshold (mins)")
    max_session_threshold = mo.ui.slider(2, 24, 1, value=24, label="Max session length (hours)")
    mo.hstack([max_session_threshold])
    return (max_session_threshold,)


@app.cell
def _(df, df_out):
    df_out.shape[0]/df.shape[0]
    return


@app.cell
def _(df, df_out, pl):
    def plot_per_date(dataf_orig, dataf_clean):
        agg_orig = (
            dataf_orig
            .with_columns(date=pl.col("datetime").dt.date())
            .group_by("date")
            .len()
            .with_columns(set=pl.lit("original"))
        )
        agg_clean = (
            dataf_clean
            .with_columns(date=pl.col("datetime").dt.date())
            .group_by("date")
            .len()
            .with_columns(set=pl.lit("clean"))
        )
        return (
            pl.concat([agg_orig, agg_clean])
            .plot
            .line(x="date", y="len", color="set")
        )

    plot_per_date(df, df_out)
    return


@app.cell(hide_code=True)
def _(pl):
    import numpy as np
    from datetime import datetime, timedelta

    def churn_dataset_generator(dataf, user_id, feature_pipeline, 
                                info_period=180, 
                                checking_period=180, 
                                start_date=datetime(2007, 1, 1), 
                                end_date=datetime(2007, 12, 31), 
                                step="1mo", 
                                time_col="datetime"):
        """
        Generates X,y pairs for churn related machine learning, with way less temporal data leaks to worry about. 

        Arguments:

        - dataf: a Polars dataframe that contains logs over time for users
        - user_id: the column name that depicts the user id
        - feature_pipeline: a Polars compatible function that generatres ML features to go in `X`
        - input_period: the number of days that the input period lasts
        - checking_period: the number of days that the checking period lasts
        - start_date: the start date for X,y-pair generation
        - end_date: the end date for X,y-pair generation
        - step: stepsize over time for new X,y-pairs. defaults to a month. 
        - time_col: column name that depicts the datetime stamp
        """
        cutoff_start = pl.datetime_range(start_date, end_date, step, eager=True).alias(time_col)
        min_date = dataf[time_col].min()
        max_date = dataf[time_col].max()

        for start in cutoff_start.to_list():
            info_period_start = start - timedelta(days=info_period)
            checking_period_end = start + timedelta(days=checking_period)
            if info_period_start < min_date:
                continue
            if checking_period_end > max_date:
                continue
            print(info_period_start, start, checking_period_end, min_date, max_date)
            train_info = dataf.filter(pl.col(time_col) < start, pl.col(time_col) >= (start - timedelta(days=info_period)))
            valid_info = dataf.filter(pl.col(time_col) >= start, pl.col(time_col) < (start + timedelta(days=checking_period)))


            target = valid_info.select("player_id").unique().with_columns(target=True)

            ml_df = (train_info
                     .pipe(feature_pipeline)
                     .join(target, on=user_id, how="left")
                     .with_columns(target=pl.when(pl.col("target")).then(True).otherwise(False)))

            X = ml_df.drop("target", "player_id")
            y = np.array(ml_df["target"]).astype(int)

            yield X, y
    return churn_dataset_generator, datetime, np


@app.cell
def _(mo):
    mo.md(
        """
    ## Stuff can go wrong!

    Now for some algorithmic details
    """
    )
    return


@app.cell
def _():
    from playtime import feats, onehot
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

    norm_feats = feats("level", "len", "spread", "max_ts") | StandardScaler()
    ohe_feats = onehot("guild", "race", "class")
    formula = norm_feats + ohe_feats
    pipe = make_pipeline(formula, LogisticRegression(max_iter=1000))
    return (
        accuracy_score,
        cross_validate,
        make_scorer,
        pipe,
        precision_score,
        recall_score,
    )


@app.cell
def _(pipe):
    pipe
    return


@app.cell
def _(pl):
    def build_sklearn_feats(dataf):
        return (
            dataf
              .group_by("player_id")
              .agg(
                  pl.len(),
                  pl.last("guild"),
                  pl.last("level"), 
                  pl.last("race"),
                  pl.last("class"), 
                  spread=pl.max("timestamp") - pl.min("timestamp"),
                  max_ts=pl.max("timestamp"),
              )
        )
    return (build_sklearn_feats,)


@app.cell
def _(build_sklearn_feats, datetime, df_out, np, pipe, pl):
    # How not to do it
    from sklearn.model_selection import train_test_split

    X_out = build_sklearn_feats(df_out)
    y_out = (
        df_out
        .with_columns(pl.col("datetime") > datetime(2008, 1, 1))
        .group_by("player_id")
        .agg(pl.max("datetime").cast(pl.Int16))["datetime"]
        .to_numpy()
    )
    X_train, X_test, y_train, y_test = train_test_split(X_out, y_out)

    np.mean(pipe.fit(X_train, y_train).predict(X_test) == y_test)
    return


@app.cell
def _(batches, pl):
    pl.concat(batches).plot.scatter(x="batch", y="test_accuracy")
    return


@app.cell(hide_code=True)
def _(
    accuracy_score,
    build_sklearn_feats,
    churn_dataset_generator,
    cross_validate,
    df_out,
    make_scorer,
    pipe,
    pl,
    precision_score,
    recall_score,
):
    gen = churn_dataset_generator(df_out, "player_id", feature_pipeline=build_sklearn_feats)

    batches = []
    for batch, (X, y) in enumerate(gen):
        scorers = {
            "accuracy": make_scorer(accuracy_score), 
            "precision": make_scorer(precision_score), 
            "recall": make_scorer(recall_score)
        }
        # Cross validate your pipeline as you might normally. Maybe even gridsearch?
        out = cross_validate(pipe, X, y, cv=5, scoring=scorers)
        batches.append(
            pl.DataFrame({k: list(v) for k, v in out.items()}).with_columns(batch=pl.lit(batch + 1))
        )
    return (batches,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

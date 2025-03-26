import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    alt.data_transformers.disable_max_rows()
    return alt, mo, pl


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


@app.cell
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
    (
        df
        .filter(pl.col("player_id").is_in(player_ids))
        .plot
        .line(x="datetime", y="level", color="player_id:N")
        .properties(title="WoW players level up over time")
    )
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
    mo.md("""
    Let's clean the dataset up and set up a proper pipeline. After all ... maybe we have some bots in here?
    """)
    return


@app.cell
def _(b, df, pl):
    def set_types(dataf):
        return (dataf.with_columns([
                    pl.col("guild").is_not_null(),
                    pl.col("datetime").cast(pl.Int64).alias("timestamp")
                ]))

    def clean_data(dataf):
        return (dataf.filter(~pl.col("class").is_in(["482", "Death Knight", "3485伊"])))

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
    b 
    def remove_bots(dataf, max_session_hours=24):
        # We're using some domain knowledge. The logger of our dataset should log
        # data every 10 minutes. That's what this line is based on.
        n_rows = max_session_hours * 6
        return (dataf
                .filter(pl.col("session_length").max().over("player_id") < n_rows))

    df_out = (
        df
        .pipe(set_types)
        .pipe(clean_data)
        .pipe(sessionize)
        .pipe(add_features)
        .pipe(remove_bots)
    )
    return add_features, clean_data, df_out, remove_bots, sessionize, set_types


@app.cell
def _(df_out):
    df_out
    return


@app.cell
def _(df, df_out):
    df_out.shape[0]/df.shape[0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

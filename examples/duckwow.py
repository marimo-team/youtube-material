# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.51.0",
#     "duckdb==1.2.2",
#     "marimo",
#     "pandas==2.2.3",
#     "pyarrow==20.0.0",
#     "sqlglot==26.17.1",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    marimo features native support for DuckDB and allows it to be used interactively together with plotting libraries that you already know and love in Python. In this notebook we will explore a dataset of world of warcraft logs that need to be sessionized. We have a dataset with a **player_id**, a **level** and a **datetime** attached. This is all we need for creating sessions with some interesting charts. 

    ## Loading in data

    Because this notebook is running in WASM it's easier to pull in the CSV file with pandas. But once we pull it in we will do some data cleaning with DuckDB so that the types are set correctly.
    """
    )
    return


@app.cell
def _(pd):
    df_start = pd.read_csv("https://raw.githubusercontent.com/koaning/wow-avatar-datasets/refs/heads/main/subset-wow.csv")
    return (df_start,)


@app.cell
def _(df_start, mo):
    df = mo.sql(
        f"""
        SELECT 
            player_id,
            level,
            CAST(datetime AS TIMESTAMP) AS datetime
        FROM df_start
        """
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This query stored it's result in a dataframe variable called `df` which can be used by polars/altair going forward. But we can also use this variable in new DuckDB queries.

    Below we are using the `table` widget from marimo to turn an aggregated dataframe into a widget that can select rows on our behalf to visualise.
    """
    )
    return


@app.cell
def _(df, mo):
    tbl = mo.ui.table(df.groupby("player_id").count().reset_index(), page_size=7, initial_selection=[2, 3])
    return (tbl,)


@app.cell
def _(alt, df, mo, tbl):
    subset = df.loc[lambda d: d["player_id"].isin(tbl.value["player_id"])]

    mo.hstack([
        tbl,
        alt.Chart(subset).mark_line().encode(x="datetime", y="level", color="player_id:N").properties(width=800)
    ], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adding sessions

    Let's now go back to DuckDB to add sessions to our dataframe.
    """
    )
    return


@app.cell
def _(mo):
    session_slider = mo.ui.slider(10, 60, 1, value=15, label="Session gap (minutes)")
    session_slider
    return (session_slider,)


@app.cell(hide_code=True)
def _(df_session, mo, session_slider):
    mo.md(f"""When we assume a session gap of {session_slider.value} minutes then there are {int(df_session["session_id"].max())} total sessions.""")
    return


@app.cell
def _(df, mo, session_slider):
    df_session = mo.sql(
        f"""
        SELECT 
            *,
            SUM(new_session) OVER (PARTITION BY player_id ORDER BY datetime) AS session_id
        FROM (
            SELECT 
                *,
                CASE 
                    WHEN datetime::TIMESTAMP - LAG(datetime::TIMESTAMP) OVER (
                        PARTITION BY player_id ORDER BY datetime
                    ) > INTERVAL '{session_slider.value} minutes' 
                    OR LAG(datetime::TIMESTAMP) OVER (PARTITION BY player_id ORDER BY datetime) IS NULL
                    THEN 1
                    ELSE 0
                END AS new_session
            FROM df
        ) subq
        ORDER BY player_id, datetime
        """,
        output=False
    )
    return (df_session,)


@app.cell
def _(mo):
    bot_slider = mo.ui.slider(1, 10, 1, label="Bot behavior threshold")

    batch = mo.md("""
    Now that we have sessions at our disposal we can try and look for sessions that suggest bot behavior. 

    {bot_slider}

    When you change this slider the stats below will update. 
    """).batch(bot_slider=bot_slider)
    batch
    return (batch,)


@app.cell
def _(batch, df_session, mo):
    n_sess = df_session.groupby(["player_id", "session_id"]).count().shape[0]
    df_bots = (
        df_session
            .groupby(["player_id", "session_id"])
            .agg(session_hrs=("level", len))
            .reset_index()
            .assign(session_hrs=lambda d: d["session_hrs"] / 6)
            [lambda d: d["session_hrs"] > batch.value["bot_slider"]]
    )

    mo.md(f"If we set the threshold of bot behavior at {batch.value["bot_slider"]} then {df_bots.shape[0] / n_sess * 100:.1f}% of all sessions should be investigated.")
    return


if __name__ == "__main__":
    app.run()

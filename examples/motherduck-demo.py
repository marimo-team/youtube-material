# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget==0.9.18",
#     "duckdb==1.2.2",
#     "marimo",
#     "polars==1.30.0",
#     "pyarrow==20.0.0",
#     "python-dotenv==1.1.0",
#     "traitlets==5.14.3",
#     "pyarrow",
#     "sqlglot",
#     "anthropic==0.51.0",
#     "pytest==8.3.5",
#     "altair==5.5.0",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    import duckdb
    import os

    conn = duckdb.connect("md:marimo-demos")
    return (conn,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1)
    slider
    return (slider,)


@app.cell(hide_code=True)
def _(ambient_air_quality, conn, mo, sample_data, slider):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            sample_data.who.ambient_air_quality
        limit {slider.value}
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Prompts and fun party tricks

    You can now run prompts for OpenAI models from motherduck, which allows for a few party tricks.
    """
    )
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        SELECT prompt('Write a poem about ducks');
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's dive into this prompt utility some more.""")
    return


@app.cell
def _(conn, mo):
    _df = mo.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS foobar
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo, movies, sample_data):
    _df = mo.sql(
        f"""
        --- Create a new table with summaries for the first 100 overview texts
        CREATE TABLE IF NOT EXISTS mydb.summary_movie AS 
            SELECT title, 
                   overview, 
                   prompt('Summarize this movie description in only four words: ' || overview) AS summary
            FROM sample_data.kaggle.movies 
            LIMIT 5;
        """,
        engine=conn
    )
    return (summary_movie,)


@app.cell
def _(conn, mo, mydb, summary_movie):
    _df = mo.sql(
        f"""
        SELECT * FROM mydb.summary_movie LIMIT 5
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo, movies, sample_data):
    _df = mo.sql(
        f"""
        SELECT
            *,
            array_cosine_similarity(overview_embeddings, title_embeddings) as sim
        FROM
            sample_data.kaggle.movies
        order by
            sim desc
        LIMIT
            100
        """,
        engine=conn
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Combining marimo features

    Let's use a slider here!
    """
    )
    return


@app.cell
def _(mo):
    session_slider = mo.ui.slider(10, 60, 1, value=15, label="Session gap (minutes)")
    session_slider
    return (session_slider,)


@app.cell
def _(conn, mo, session_slider, wowfull):
    df_sess = mo.sql(
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
            FROM wowfull LIMIT 1000
        ) subq
        ORDER BY player_id, datetime
        """,
        engine=conn
    )
    return (df_sess,)


@app.cell
def _():
    import pytest
    return


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(df_sess, pl):
    def test_session_starts_at_1():
        first_ids = [i for i in df_sess.head(100).group_by("player_id").first()["session_id"]]
        for i in first_ids:
            assert i == 1


    def test_level_increases_over_player_id():
        out = (
            df_sess.head(100)
            .with_columns(pl.col("level").diff().over("player_id").alias("diff"))
            .select("diff")
            .drop_nans()
            .filter(pl.col("diff") < 0)
        )
        assert out.shape[0] == 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's now calculate the amount of time that is spent getting to level 60.""")
    return


@app.cell
def _(mo):
    dropdown = mo.ui.dropdown(options=["race", "in_guild", "class"], value="race")
    dropdown
    return (dropdown,)


@app.cell(hide_code=True)
def _(conn, dropdown, mo, mydb, player_sessions):
    df_level = mo.sql(
        f"""
        WITH GUILDED AS (
            SELECT *,
            CASE WHEN guild IS NOT NULL THEN 1 ELSE 0 END AS in_guild
          FROM mydb.player_sessions
          WHERE class IN (
            'Druid', 'Hunter', 'Mage', 'Paladin', 'Priest', 'Rogue', 'Warlock', 'Warrior'
          ) AND race IN (
            'Undead', 'Troll', 'Tauren', 'Orc', 'Blood Elf'
          )
        ),

        FOO AS (
          SELECT 
            count(*) as count, 
            count(DISTINCT player_id) as player_count,
            count(*) / count(DISTINCT player_id) / 6 as average_hours_played,
            level, 
            {dropdown.value}
          FROM GUILDED
          WHERE level < 60 AND level > 1
          GROUP BY level, {dropdown.value}
        )

        SELECT * FROM FOO
        """,
        output=False,
        engine=conn
    )
    return (df_level,)


@app.cell
def _(df_level, dropdown):
    df_level.plot.line("level", "average_hours_played", color=f"{dropdown.value}:N")
    return


@app.cell
def _(df_level, dropdown, pl):
    df_level.group_by(dropdown.value).agg(
        pl.col("average_hours_played").sum().alias("average_total_hours")
    ).sort("average_total_hours", descending=True)
    return


if __name__ == "__main__":
    app.run()

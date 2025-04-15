import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sqlalchemy

    DATABASE_URL = "sqlite:///sqlite.db"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return DATABASE_URL, engine, sqlalchemy


@app.cell
def _(engine, mo, pokemon):
    df_poke = mo.sql(
        f"""
        SELECT hp, kind, attack FROM pokemon;
        """,
        output=False,
        engine=engine
    )
    return (df_poke,)


@app.cell
def _(df_poke):
    df_edit = df_poke.head(100)
    return (df_edit,)


@app.cell
def _(df_edit, mo):
    dropdown = mo.ui.dropdown(options=df_edit["kind"].unique())
    dropdown
    return (dropdown,)


@app.cell
def _(df_edit, dropdown, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM df_edit WHERE kind = '{dropdown.value}' ;
        """
    )
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        select * from read_parquet("path/to/file.parquet")
        """
    )
    return


if __name__ == "__main__":
    app.run()

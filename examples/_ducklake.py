import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's start by installing the requirements. Note you must have `duckdb==1.3` installed for this to work.""")
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        INSTALL ducklake;
        ATTACH 'ducklake:metadata.ducklake' AS my_ducklake;
        """
    )
    return (my_ducklake,)


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        CREATE OR REPLACE TABLE my_ducklake.demo (i INTEGER);
        """
    )
    return (demo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's play around a little bit by inserting some values into a new table.""")
    return


@app.cell
def _():
    import random

    i = random.randint(1, 1000)
    i
    return i, random


@app.cell
def _(demo, i, mo, my_ducklake):
    _df = mo.sql(
        f"""
        INSERT INTO my_ducklake.demo VALUES ({i});
        """
    )
    return


@app.cell
def _(demo, mo, my_ducklake, random):
    import time 

    for _ in range(10): 
        time.sleep(0.1)
        mo.sql(f"INSERT INTO my_ducklake.demo VALUES ({random.randint(1, 1000)});")
    return


@app.cell
def _(demo, mo, my_ducklake):
    _df = mo.sql(
        f"""
        SELECT * FROM my_ducklake.demo;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""But here's the fun bit. All those inserts ... those snapshots are tracked!""")
    return


@app.cell
def _(mo):
    versions = mo.sql(
        f"""
        FROM ducklake_snapshots('my_ducklake');
        """
    )
    return


@app.cell
def _(mo):
    version = mo.ui.slider(1, 30, label="Time travel")
    version
    return


app._unparsable_cell(
    r"""
    FROM my_ducklake.demo AT (VERSION => {version.value});
    """,
    name="_"
)


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_table_stats, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_table_stats LIMIT 100
        """
    )
    return


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_snapshot, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_snapshot LIMIT 100
        """
    )
    return


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_metadata, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_metadata LIMIT 100
        """
    )
    return


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_inlined_data_tables, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_inlined_data_tables LIMIT 100
        """
    )
    return


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_file_column_statistics, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_file_column_statistics LIMIT 100
        """
    )
    return


@app.cell
def _(__ducklake_metadata_my_ducklake, ducklake_table, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM __ducklake_metadata_my_ducklake.ducklake_table LIMIT 100
        """
    )
    return


if __name__ == "__main__":
    app.run()

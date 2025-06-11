# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "anthropic==0.52.2",
#     "marimo",
#     "memvid==0.1.3",
#     "pandas==2.3.0",
#     "polars==1.30.0",
#     "pyarrow==20.0.0",
#     "pyiceberg==0.9.1",
#     "python-dotenv==1.1.0",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that you run this command beforehand: 

    ```
    npx wrangler r2 bucket catalog enable <bucket-name>
    ```
    """
    )
    return


@app.cell
def _():
    import altair as alt
    return


@app.cell
def _():
    import polars as pl
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import os
    from dotenv import load_dotenv
    from pyiceberg.catalog.rest import RestCatalog

    load_dotenv(".env")

    # Define catalog connection details (replace variables)
    WAREHOUSE = os.environ.get("R2_WAREHOUSE")
    TOKEN = os.environ.get("R2_TOKEN")
    CATALOG_URI = os.environ.get("R2_CATALOG_URI")

    # Connect to R2 Data Catalog
    catalog = RestCatalog(
        name="my_catalog",
        warehouse=WAREHOUSE,
        uri=CATALOG_URI,
        token=TOKEN,
    )
    return catalog, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we've made a connection, we can start doing moving some data in and out.""")
    return


@app.cell
def _(catalog):
    # Create default namespace if needed
    catalog.create_namespace_if_not_exists("default")
    return


@app.cell
def _(pl):
    # Create simple PyArrow table
    df = (
        pl.read_csv("services-2024.csv.gz")
            .with_columns(pl.col("Service:Date").str.strptime(pl.Date, "%Y-%m-%d"))
    ).to_arrow()
    return (df,)


@app.cell
def _(df):
    df.schema
    return


@app.cell
def _(catalog, df):
    # Create or load Iceberg table
    test_table = ("default", "train")

    if not catalog.table_exists(test_table):
        print(f"Creating table: {test_table}")
        table = catalog.create_table(
            test_table,
            schema=df.schema,
        )

        table.append(df)
    return


@app.cell
def _(catalog, pl):
    (
        catalog
            .load_table("default.train")
            .to_polars()
            .filter(pl.col("Service:Date") == pl.datetime(2024, 1, 1))
            .collect()
    )
    return


app._unparsable_cell(
    r"""
    from pyiceberg.expressions import EqualTo

    (
        catalog
            .load_table(\"default.train\")
            .scan(row_filter=EqualTo(\"Service:Date\", \"2024-01-01\"))
            .to_polars()
    )d
    """,
    name="_"
)


@app.cell
def _():
    # Optional cleanup. To run uncomment and run cell
    # print(f"Deleting table: {test_table}")
    # catalog.drop_table(test_table)
    # print("Table dropped.")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

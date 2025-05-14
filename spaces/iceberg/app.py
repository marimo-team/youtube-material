# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "polars==1.29.0",
#     "pyarrow==20.0.0",
#     "pyiceberg==0.9.1",
#     "sqlalchemy==2.0.40",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sqlalchemy
    import polars as pl
    from pathlib import Path
    from pyiceberg.partitioning import PartitionSpec, PartitionField
    from pyiceberg.transforms import IdentityTransform
    return IdentityTransform, PartitionField, PartitionSpec, mo, pl


@app.cell
def _():
    from pyiceberg.catalog import load_catalog

    warehouse_path = "warehouse"
    catalog = load_catalog(
        "default",
        **{
            'type': 'sql',
            "uri": f"sqlite:///{warehouse_path}/iceberg.db",
            "warehouse": f"file://{warehouse_path}",
        },
    )
    return (catalog,)


@app.cell
def _(pl):
    df_taxi = pl.read_csv("yellow_tripdata_2015-01.csv").to_arrow()
    return (df_taxi,)


@app.cell
def _(df_taxi):
    df_taxi.group_by("passenger_count").aggregate([([], "count_all")])
    return


@app.cell
def _(IdentityTransform, PartitionField, PartitionSpec):
    spec = PartitionSpec(
        PartitionField(source_id=3, field_id=1000, name="passenger_count", transform=IdentityTransform())
    )
    return


@app.cell
def _(df_taxi):
    df_taxi.schema
    return


@app.cell
def _(catalog, df_taxi):
    catalog.create_namespace_if_not_exists("default")

    table = catalog.create_table_if_not_exists(
        "default.taxi",
        schema=df_taxi.schema,
    )
    return (table,)


@app.cell
def _(df_taxi, table):
    if not table.current_snapshot():
        table.append(df_taxi)
    return


@app.cell
def _(catalog):
    (
        catalog
            .load_table("default.taxi")
            .to_polars()
            .group_by("passenger_count")
            .len()
            .sort("passenger_count")
            .collect()
    )
    return


@app.cell
def _(pl):
    pl.scan_csv("yellow_tripdata_2015-01.csv").group_by("passenger_count").len().sort("passenger_count").collect()
    return


@app.cell
def _(pl):
    pl.read_csv("yellow_tripdata_2015-01.csv").group_by("passenger_count").len().sort("passenger_count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The partition is great, but the comparison with `read_csv` is a bit unfair. Let's convert the `.csv` file to `.parquet` and also add a partition in polars with statistics. """)
    return


@app.cell
def _(pl):
    pl.read_csv("yellow_tripdata_2015-01.csv").write_parquet("taxi.parquet", partition_by=["passenger_count"], statistics=True)
    return


@app.cell
def _(pl):
    pl.scan_parquet("taxi.parquet").group_by("passenger_count").len().sort("passenger_count").collect()
    return


@app.cell
def _(pl):
    pl.read_parquet("taxi.parquet").group_by("passenger_count").len().sort("passenger_count")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

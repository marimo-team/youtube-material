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

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sqlalchemy
    import polars as pl
    from pathlib import Path
    from pyiceberg.partitioning import PartitionSpec, PartitionField
    from pyiceberg.transforms import IdentityTransform
    from zipfile import ZipFile
    return (
        IdentityTransform,
        PartitionField,
        PartitionSpec,
        Path,
        ZipFile,
        mo,
        pl,
    )


@app.cell
def _(Path):
    from pyiceberg.catalog import load_catalog

    Path("warehouse").mkdir(exist_ok=True, parents=True)

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
def _(ZipFile, pl):
    df_orig = pl.read_csv(ZipFile("yellow_tripdata_2015-01.csv.zip").open("yellow_tripdata_2015-01.csv").read())
    df_taxi = df_orig.to_arrow()
    return df_orig, df_taxi


@app.cell
def _(df_taxi):
    df_taxi.group_by("passenger_count").aggregate([([], "count_all")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's now take this pyarrow dataframe and prepare it for insertion. We want to extract the right schema and also add a partition. """)
    return


@app.cell
def _(df_taxi):
    import pyarrow as pa
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        NestedField, IntegerType, StringType, DoubleType, TimestampType
    )
    from pyiceberg.table.name_mapping import NameMapping, MappedField
    from pyiceberg.io.pyarrow import pyarrow_to_schema

    # Create a mapping from column names to field IDs
    name_mapping_fields = []
    for idx, field_name in enumerate(df_taxi.column_names, start=1):
        name_mapping_fields.append(MappedField(field_id=idx, names=[field_name]))

    # Create a name mapping
    name_mapping = NameMapping(name_mapping_fields)

    # Convert PyArrow schema to Iceberg schema
    iceberg_schema = pyarrow_to_schema(df_taxi.schema, name_mapping)

    # Now find the field ID for 'passenger_count'
    passenger_count_field = iceberg_schema.find_field("passenger_count")
    source_id = passenger_count_field.field_id

    print(f"The source_id for 'passenger_count' is: {source_id}")
    return (iceberg_schema,)


@app.cell
def _(IdentityTransform, PartitionField, PartitionSpec):
    spec = PartitionSpec(
        PartitionField(
            source_id=4, field_id=1000, name="passenger_count", transform=IdentityTransform())
    )
    return (spec,)


@app.cell
def _(catalog, iceberg_schema, spec):
    catalog.create_namespace_if_not_exists("default")

    table = catalog.create_table_if_not_exists(
        "default.taxi",
        schema=iceberg_schema,
        partition_spec=spec
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's write the original zipped file into a csv file. We can read this and perform the same query to compare speeds.""")
    return


@app.cell
def _(df_orig):
    df_orig.write_csv("taxi.csv")
    return


@app.cell
def _(pl):
    pl.scan_csv("taxi.csv").group_by("passenger_count").len().sort("passenger_count").collect()
    return


@app.cell
def _(pl):
    pl.read_csv("taxi.csv").group_by("passenger_count").len().sort("passenger_count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    That's a bunch slower!

    A part of the reason is that iceberg had partitions in it, which is great, but the comparison with `read_csv` is a bit unfair. Let's convert the `.csv` file to `.parquet` and also add a partition in polars with statistics. You will now see that we get a similar performance.
    """
    )
    return


@app.cell
def _(df_orig):
    df_orig.write_parquet("taxi.parquet", partition_by=["passenger_count"], statistics=True)
    return


@app.cell
def _(pl):
    pl.scan_parquet("taxi.parquet").group_by("passenger_count").len().sort("passenger_count").collect()
    return


@app.cell
def _(pl):
    pl.read_parquet("taxi.parquet").group_by("passenger_count").len().sort("passenger_count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""So keep in mind that polars can for sure also speed things up if you are aware of what you are doing. But one nice thing about iceberg is that can be seen as a catalogue with *a bunch* of good habbits for performance later down the line.""")
    return


if __name__ == "__main__":
    app.run()

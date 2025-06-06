# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "duckdb==1.2.2",
#     "anthropic==0.49.0",
#     "sqlglot==26.14.0",
#     "polars[pyarrow]==1.27.1",
#     "altair==5.5.0",
#     "pytest==8.3.5",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(duckdb, initialize_database):
    from pathlib import Path

    file_map = {_.stem.replace("_dataset", ""): str(_) for _ in Path("brazil-webshop").glob("*")}

    if not Path("webshop.db").exists():
        initialize_database(db_path="webshop.db", csv_mappings=file_map)

    DATABASE_URL = "webshop.db"
    engine = duckdb.connect(DATABASE_URL, read_only=True)
    return (engine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Ecommerce dataset

    Let's explore DuckDB by exploring [this dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download).

    ![](https://i.imgur.com/HRhd2Y0.png)
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import duckdb

    def initialize_database(
        db_path: str,
        csv_mappings: dict[str, str],
        overwrite: bool = False
    ) -> None:
        """
        Create a persistent DuckDB database and load CSV files into tables.

        Args:
            db_path: Path where the DuckDB file should be created/loaded
            csv_mappings: Dictionary mapping table names to CSV file paths
            overwrite: If True, will drop existing tables before creating new ones

        Example:
            mappings = {
                "orders": "path/to/orders.csv",
                "customers": "path/to/customers.csv",
                "products": "path/to/products.csv"
            }
            initialize_database("my_store.db", mappings)
        """
        # Connect to a persistent database
        conn = duckdb.connect(db_path)

        for table_name, csv_path in csv_mappings.items():
            if overwrite:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table from CSV
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_path}')
            """)

        # Optional: Create indexes or views here if needed

        conn.close()
    return duckdb, initialize_database


@app.cell
def _():
    import polars as pl
    return


@app.cell
def _(engine, mo, olist_orders):
    pltr = mo.sql(
        f"""
        SELECT 
            order_id,
            order_status,
            order_approved_at,
            order_estimated_delivery_date,
            DATE_DIFF('day', order_approved_at, order_estimated_delivery_date) as estimated_delivery_days,
            DATE_DIFF('day', order_approved_at, order_delivered_customer_date) as actual_delivery_days,
            CASE 
                WHEN DATE_DIFF('day', order_delivered_customer_date, order_estimated_delivery_date) > 0 
                THEN 'Early'
                WHEN DATE_DIFF('day', order_delivered_customer_date, order_estimated_delivery_date) < 0 
                THEN 'Late'
                ELSE 'On Time'
            END as delivery_status
        FROM olist_orders
        WHERE order_status = 'delivered'
            AND order_approved_at IS NOT NULL
            AND order_delivered_customer_date IS NOT NULL
        """,
        engine=engine
    )
    return (pltr,)


@app.cell
def _(pltr):
    import altair as alt

    # Create a histogram of estimated vs actual delivery times
    base = alt.Chart(pltr.head(10_000)).encode(
        alt.X('estimated_delivery_days', bin=alt.Bin(maxbins=30), title='Days'),
        alt.Y('count()', title='Number of Orders'),
        tooltip=[
            alt.Tooltip('estimated_delivery_days', title='Estimated Days'),
            alt.Tooltip('count()', title='Number of Orders')
        ]
    )

    histogram = base.mark_bar(color='#5276A7', opacity=0.5).encode(
        alt.X('estimated_delivery_days', title='Delivery Time (Days)'),
    ) + base.mark_bar(color='#F18727', opacity=0.5).encode(
        alt.X('actual_delivery_days', title='Delivery Time (Days)'),
    )

    # Add a legend
    legend = alt.Chart({
        'values': [
            {'category': 'Estimated', 'color': '#5276A7'},
            {'category': 'Actual', 'color': '#F18727'}
        ]
    }).mark_rect().encode(
        alt.X('category:N', title=None),
        alt.Color('color:N', scale=None)
    )

    # Combine the charts
    final_chart = (histogram | legend).properties(
        title='Distribution of Estimated vs Actual Delivery Times',
        width=600,
        height=400
    )

    final_chart
    return (alt,)


@app.cell
def _(engine, mo, olist_orders):
    df_time = mo.sql(
        f"""
        WITH daily_deliveries AS (
            SELECT 
                DATE_TRUNC('day', order_delivered_customer_date) as delivery_date,
                COUNT(*) as total_deliveries,
                COUNT(CASE WHEN DATE_DIFF('day', order_delivered_customer_date, order_estimated_delivery_date) > 0 THEN 1 END) as early_deliveries,
                COUNT(CASE WHEN DATE_DIFF('day', order_delivered_customer_date, order_estimated_delivery_date) = 0 THEN 1 END) as ontime_deliveries,
                COUNT(CASE WHEN DATE_DIFF('day', order_delivered_customer_date, order_estimated_delivery_date) < 0 THEN 1 END) as late_deliveries
            FROM olist_orders
            WHERE 
                order_status = 'delivered'
                AND order_delivered_customer_date IS NOT NULL
                AND order_estimated_delivery_date IS NOT NULL
            GROUP BY DATE_TRUNC('day', order_delivered_customer_date)
        )
        SELECT 
            delivery_date,
            total_deliveries,
            ROUND(100.0 * early_deliveries / total_deliveries, 2) as pct_early,
            ROUND(100.0 * ontime_deliveries / total_deliveries, 2) as pct_ontime,
            ROUND(100.0 * late_deliveries / total_deliveries, 2) as pct_late
        FROM daily_deliveries
        ORDER BY delivery_date;
        """,
        engine=engine
    )
    return (df_time,)


@app.cell
def _(alt, df_time):
    alt.Chart(df_time).mark_line().encode(x="delivery_date", y="pct_late")
    return


@app.cell
def _(df_time, engine, mo):
    def test_unique_dates():
        out = mo.sql(
            """
            SELECT * FROM df_time
            """,
            engine=engine
        )
        assert out["delivery_date"].n_unique() == out.shape[0]
    return


@app.cell
def _(df_time, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM df_time
        """
    )
    return


if __name__ == "__main__":
    app.run()

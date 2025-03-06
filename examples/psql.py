import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## SQL variation in DuckDB

    There are a lot of ways to write SQL in DuckDB. Below is a 'standard' way of doing which typically results in a nested SQL command that can be hard to read
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT
            i.customer_id,
            c.last_name || ', ' || c.first_name AS name,
            SUM(i.total - 0.8) AS sum_income,
            version() AS db_version
        FROM (
            SELECT
                *,
                0.8 AS transaction_fees,
                total - 0.8 AS income
            FROM 'https://raw.d.com/ywelsch/duckdb-psql/main/example/invoices.csv'
            WHERE invoice_date >= DATE '1970-01-16' AND (total - 0.8) > 1
        ) i
        JOIN 'https://raw.githubusercontent.com/ywelsch/duckdb-psql/main/example/customers.csv' c
            ON i.customer_id = c.customer_id
        GROUP BY i.customer_id, c.last_name, c.first_name
        ORDER BY sum_income DESC
        LIMIT 10;
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    DuckDB does come with some helpers though. Instead of nesting everything you can also generate tables that act as milestones that are easy to reason about. These are called "common table expressions" and they kind of allow you to think of tables as variables in Python. Instead of one big nested query, you are able to split things up a little.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH filtered_invoices AS (
            SELECT
                *,
                0.8 AS transaction_fees,
                total - 0.8 AS income
            FROM 'https://raw.githubusercontent.com/ywelsch/duckdb-psql/main/example/invoices.csv'
            WHERE invoice_date >= DATE '1970-01-16' AND (total - 0.8) > 1
        ),
        aggregated_invoices AS (
            SELECT
                customer_id,
                AVG(total) AS avg_total,
                SUM(income) AS sum_income,
                COUNT(*) AS ct
            FROM filtered_invoices
            GROUP BY customer_id
            ORDER BY sum_income DESC
            LIMIT 10
        )
        SELECT
            a.customer_id,
            c.last_name || ', ' || c.first_name AS name,
            a.sum_income,
            version() AS db_version
        FROM aggregated_invoices a
        JOIN 'https://raw.githubusercontent.com/ywelsch/duckdb-psql/main/example/customers.csv' c
            ON a.customer_id = c.customer_id;
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    But you can also go even a step further by using PSQL. This is a variation of SQL that introduces a pipe symbol `|>` that can be interpreted as a `then` statement. Do a thing `then` do something else. This allows for a style of SQL that is very similar to method chaining that you might be used to from dataframe libraries in Python.""")
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        install psql from community;
        load psql;

        from 'https://raw.githubusercontent.com/ywelsch/duckdb-psql/main/example/invoices.csv' |>
        where invoice_date >= date '1970-01-16' |>
        select
          *, 
          0.8 as transaction_fees,
          total - transaction_fees as income |>
        where income > 1 |>
        select
          customer_id, 
          avg(total), 
          sum(income) as sum_income, 
          count() as ct
          group by customer_id |>
        order by sum_income desc |>
        limit 10 |>
        as invoices
          join 'https://raw.githubusercontent.com/ywelsch/duckdb-psql/main/example/customers.csv'
            as customers
          on invoices.customer_id = customers.customer_id |>
        select
          customer_id,
          last_name || ', ' || first_name as name,
          sum_income,
          version() as db_version;

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

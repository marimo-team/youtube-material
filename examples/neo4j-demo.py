# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.51.0",
#     "k==0.0.1",
#     "marimo",
#     "neo4j==5.28.1",
#     "polars==1.29.0",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(".env")
    return GraphDatabase, os


@app.cell
def _(GraphDatabase, os):
    # Replace with the actual URI, username and password
    AURA_CONNECTION_URI = "neo4j+s://9ba2dd61.databases.neo4j.io"
    AURA_USERNAME = "neo4j"
    AURA_PASSWORD = os.environ.get("NEO4J_PASSWORD")

    # Driver instantiation
    driver = GraphDatabase.driver(AURA_CONNECTION_URI, auth=(AURA_USERNAME, AURA_PASSWORD))
    return (driver,)


@app.cell
def _(driver):
    driver
    return


@app.cell
def _(driver):
    driver.verify_connectivity()
    return


@app.cell
def _():
    return


@app.cell
def _():
    import polars as pl 

    projects = pl.read_csv("dependencies.csv").group_by("project").len()["project"].to_list()
    return pl, projects


@app.cell
def _(driver, mo, projects):
    for name in mo.status.progress_bar(projects):
        driver.execute_query("""
        MERGE (p:Project {name: $name})
        """, name=name)
    return


@app.cell
def _(driver, mo, pl):
    for link in mo.status.progress_bar(pl.read_csv("dependencies.csv").to_dicts()):
        if link["required"]:
            driver.execute_query("""
            MERGE (p1:Project {name: $name1})-[:depends]->(p2:Project {name: $name2})
            """, name1=link["dep"], name2=link["project"])
    return


@app.cell
def _(driver):
    records, summary, keys = driver.execute_query(
        "MATCH path = (p1:Project {name: $name})-[*1..4]->(p2) RETURN p1.name, p2.name, path",
        name="pandas", 
    )
    return (records,)


@app.cell
def _(records):
    records
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

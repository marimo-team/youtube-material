import marimo

__generated_with = "0.13.11"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        INSTALL ducklake;
        ATTACH 'ducklake:metadata.ducklake' AS myducklake;
        """
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, mo, pl


@app.cell
def _(pl):
    df_cars = pl.read_csv("https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv")
    return (df_cars,)


@app.cell
def _(df_cars):
    df_cars.head()
    return


@app.cell
def _(df_cars):
    df_cars.plot.scatter(x="mpg", y="hp")
    return


if __name__ == "__main__":
    app.run()

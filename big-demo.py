import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import altair as alt
    import numpy as np
    from mohtml import br
    return alt, br, mo, np, pd, pl


@app.cell
def _(mo):
    mo.md(
        """
        ## Exploratory data analysis

        With marimo it is pretty easy to whip together a small UI for exploratory data analysis. Below is a small example for a use-case that relates to solar panel generation.

        First, let's show how altair charts are query-able in marimo.
        """
    )
    return


@app.cell
def _(pl):
    df_meteo = (
        pl.read_csv("https://raw.githubusercontent.com/koaning/solah/refs/heads/main/data/history.csv")
        .with_columns(
            date=pl.col("date").str.to_date(), 
        )
    )
    return (df_meteo,)


@app.cell
def _(alt, mo, pl):
    df_generated = (
        pl.read_csv("https://raw.githubusercontent.com/koaning/solah/refs/heads/main/data/generated.csv")
            .with_columns(
                date=pl.col("date").str.to_date(format="%m/%d/%Y"), 
                kWh=pl.col("kWh").str.replace(",", "").cast(pl.Int32)/1000
            )
    )

    alt_chart = alt.Chart(df_generated).mark_point().encode(
        x="date:T",
        y="kWh:Q"
    ).properties(
        width=600,
        height=300
    )

    chart = mo.ui.altair_chart(alt_chart, chart_selection=True)
    return alt_chart, chart, df_generated


@app.cell
def _(chart, mo):
    mo.hstack([
        chart, 
        chart.value.head(5)
    ])
    return


@app.cell
def _(mo):
    mo.md("""Next, let's show how you can use marimo's input elements to easily make an interactive interface to explore the ML features.""")
    return


@app.cell
def _(br):
    br()
    return


@app.cell
def _(df_meteo, mo):
    cols = [n for n in df_meteo.columns if n != "date"]

    radio_col = mo.ui.radio(options=cols, value="sunshine_duration")
    return cols, radio_col


@app.cell
def _(df_generated, df_meteo):
    df_merged = df_generated.join(df_meteo, left_on="date", right_on="date").drop_nulls()
    return (df_merged,)


@app.cell
def _(df_merged, mo, radio_col):
    mo.hstack([
        radio_col, 
        df_merged.plot.scatter("date", radio_col.value), 
        df_merged.plot.scatter(radio_col.value, "kWh")
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Interactive explainers

        Marimo features many input elements, but it also benefits from a small ecosystem of plugins via the anywidget integration. Below are some examples from the `wigglystuff` library that help highlight how expressive the library can be.
        """
    )
    return


@app.cell
def _(alt, np, pd, prob1, prob2):
    cores = np.arange(1, 64 + 1)
    p1, p2 = prob1.amount / 100, prob2.amount / 100
    eff1 = 1 / (p1 + (1 - p1) / cores)
    eff2 = 1 / (p2 + (1 - p2) / cores)

    df_amdahl = pd.DataFrame(
        {
            "cores": cores,
            f"{prob1.amount:.2f}% sync rate": eff1,
            f"{prob2.amount:.2f}% sync rate": eff2,
        }
    ).melt("cores")

    c = (
        alt.Chart(df_amdahl)
        .mark_line()
        .encode(
            x="cores", y=alt.Y("value").title("effective cores"), color="variable"
        )
        .properties(
            width=500, title="Comparison between cores and actual speedup."
        )
    )
    return c, cores, df_amdahl, eff1, eff2, p1, p2


@app.cell
def _(mo):
    from wigglystuff import TangleSlider, TangleChoice

    coffees = mo.ui.anywidget(
        TangleSlider(amount=10, min_value=0, step=1, suffix=" coffees", digits=0)
    )
    price = mo.ui.anywidget(
        TangleSlider(
            amount=3.50,
            min_value=0.01,
            max_value=10,
            step=0.01,
            prefix="$",
            digits=2,
        )
    )
    prob1 = mo.ui.anywidget(
        TangleSlider(
            min_value=0, max_value=20, step=0.1, suffix="% of the time", amount=5
        )
    )
    prob2 = mo.ui.anywidget(
        TangleSlider(
            min_value=0, max_value=20, step=0.1, suffix="% of the time", amount=0
        )
    )
    saying = mo.ui.anywidget(TangleChoice(["🙂", "🎉", "💥"]))
    times = mo.ui.anywidget(
        TangleSlider(min_value=1, max_value=20, step=1, suffix=" times", amount=3)
    )
    return (
        TangleChoice,
        TangleSlider,
        coffees,
        price,
        prob1,
        prob2,
        saying,
        times,
    )


@app.cell
def _(c, coffees, mo, price, prob1, prob2, saying, times, total):
    mo.vstack(
        [
            mo.md(
                f"""
        ### Apples example 

        Suppose that you have {coffees} and they each cost {price} then in total you would need to spend ${total:.2f}. 

        ### Amdhals law

        You cannot always get a speedup by throwing more compute at a problem. Let's compare two scenarios. 

        - You might have a parallel program that needs to sync up {prob1}.
        - Another parallel program needs to sync up {prob2}.

        The consequences of these choices are shown below. You might be suprised at the result, but you need to remember that if you throw more cores at the problem then you will also have more cores that will idle when the program needs to sync. 

        """
            ),
            c,
            mo.md(
                f"""
        ### Also a choice widget 

        As a quick demo, let's repeat {saying} {times}. 

        {" ".join([saying.choice] * int(times.amount))}
        """
            ),
        ]
    )
    return


@app.cell
def _(coffees, price):
    # You need to define derivates in other cells.
    total = coffees.amount * price.amount
    return (total,)


@app.cell
def _():
    from wigglystuff import Matrix
    return (Matrix,)


@app.cell
def _(alt, color, mo, pca_mat, pd, rgb_mat):
    X_tfm = rgb_mat @ pca_mat.matrix
    df_pca = pd.DataFrame({"x": X_tfm[:, 0], "y": X_tfm[:, 1], "c": color})
    pca_chart = alt.Chart(df_pca).mark_point().encode(x="x", y="y", color=alt.Color('c:N', scale = None))

    mo.vstack([
        mo.md("""
    ### PCA demo with `Matrix` 

    Ever want to do your own PCA? Try to figure out a mapping from a 3d color map to a 2d representation with the transformation matrix below."""),
        mo.hstack([pca_mat, pca_chart])
    ])
    return X_tfm, df_pca, pca_chart


@app.cell
def _(Matrix, mo, np, pd):
    pca_mat = mo.ui.anywidget(Matrix(np.random.normal(0, 1, size=(3, 2)), step=0.1))
    rgb_mat = np.random.randint(0, 255, size=(1000, 3))
    color = ["#{0:02x}{1:02x}{2:02x}".format(r, g, b) for r,g,b in rgb_mat]

    rgb_df = pd.DataFrame({
        "r": rgb_mat[:, 0], "g": rgb_mat[:, 1], "b": rgb_mat[:, 2], 'color': color
    })
    return color, pca_mat, rgb_df, rgb_mat


@app.cell
def _(mo):
    mo.md(
        """
        ## Hiplot demo

        To help give a glimpse of some future work, let's consider what it could be like to tackle the famous `creditcard` dataset with interactive tooling. There is no tight integration with hiplot just yet, but it still works because marimo can leverage it's HTML tools to still render things in the notebook.
        """
    )
    return


@app.cell
def _(pl):
    from sklearn.datasets import fetch_openml

    df_credit = fetch_openml(
        data_id=1597,
        as_frame=True
    )

    df_credit = df_credit['frame'].rename(columns={"Class": "group"})
    df_credit['group'] = df_credit['group'] == '1'
    df_credit = pl.DataFrame(df_credit)
    return df_credit, fetch_openml


@app.cell
def _():
    import hiplot as hip
    return (hip,)


@app.cell
def _(df_credit, pl):
    json_data = pl.concat([
        df_credit.filter(~pl.col("group")).sample(10_000),
        df_credit.filter(pl.col("group")),
    ]).drop("Amount").to_dicts()
    return (json_data,)


@app.cell
def _(hip, json_data, mo):
    mo.iframe(hip.Experiment.from_iterable(json_data).to_html(), height="700px")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

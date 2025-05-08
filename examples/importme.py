

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="full")

with app.setup:
    import polars as pl


app._unparsable_cell(
    r"""
    # Initialization code that runs before all other cells
    import marimo as mo
    import random
    import numpy as np
    import altair as altk@
    """,
    name="_"
)


@app.function
def apply_moving_average(dataf, window_size=12):
    return dataf.with_columns(
        pl.col("cumsum")
        .rolling_mean(window_size=window_size, center=True, min_samples=1)
        .alias("smoothed")
    )


@app.cell
def _(create_chart, mo, smoothed_df, window_size_slider):
    mo.vstack([
        window_size_slider,
        create_chart(smoothed_df, window_size_slider.value)
    ])
    return


@app.cell
def _(np, random):
    # Set a random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate random data
    n_points = 500
    random_values = [random.uniform(-1, 1.1) for _ in range(n_points)]
    cumsum_values = np.cumsum(random_values)

    # Create a dictionary with the data
    data_dict = {
        "index": list(range(n_points)),
        "random_value": random_values,
        "cumsum": cumsum_values
    }
    return (data_dict,)


@app.cell
def _(mo):
    # Create a slider for window size
    window_size_slider = mo.ui.slider(
        start=2, 
        stop=50, 
        value=10, 
        step=1,
        label="Moving Average Window Size"
    )
    return (window_size_slider,)


@app.cell
def _(data_dict):
    # Create a Polars DataFrame
    df = pl.DataFrame(data_dict)
    return (df,)


@app.cell
def _(alt):
    def create_chart(df, window_size):
        # Prepare data for visualization
        # Convert to long format for Altair using unpivot instead of melt
        plot_df = df.select(
            ["index", "cumsum", "smoothed"]
        ).unpivot(
            index=["index"], 
            on=["cumsum", "smoothed"],
            variable_name="series",
            value_name="value"
        )

        # Create the chart
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('index:Q', title='Time'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('series:N', 
                           legend=alt.Legend(title="Series"),
                           scale=alt.Scale(
                               domain=['cumsum', 'smoothed'],
                               range=['#1f77b4', '#ff7f0e']
                           )),
            tooltip=['index:Q', 'value:Q', 'series:N']
        ).properties(
            width=700,
            height=400,
            title=f'Original vs Smoothed Data (Window Size: {window_size})'
        ).interactive()

        return chart
    return (create_chart,)


@app.cell
def _(df, window_size_slider):
    smoothed_df = df.pipe(apply_moving_average, window_size_slider.value)
    return (smoothed_df,)


if __name__ == "__main__":
    app.run()

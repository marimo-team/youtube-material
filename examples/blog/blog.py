import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1, label="Demo slider")
    slider
    return (slider,)


@app.cell
def _(slider):
    slider.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

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
    return (slider,)


@app.cell
def _(mo, slider):
    mo.md(
        f"""
        marimo is a **reactive** Python notebook.

        This means that unlike traditional notebooks, marimo notebooks **run
        automatically** when you modify them or
        interact with UI elements, like this slider: {slider}.

        {"##" + "🍃" * slider.value}
        """
    )
    return


if __name__ == "__main__":
    app.run()

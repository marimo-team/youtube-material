import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    slider = mo.ui.slider(0, 100, 1, label="slider input")
    text = mo.ui.text(label="text input")
    text_area = mo.ui.text_area(label="large text input")
    return slider, text, text_area


@app.cell
def _(mo, slider, text, text_area):
    mo.vstack([
        slider, 
        text,
        text_area
    ])
    return


@app.cell
def _(slider):
    slider.value
    return


@app.cell
def _(mo, slider, text):
    md = mo.md("""This is a simple form. 

    {slider}

    {text}
    """
    )
    form = md.batch(slider=slider, text=text).form()
    return form, md


@app.cell
def _(form):
    form
    return


@app.cell
def _(form):
    form.value
    return


if __name__ == "__main__":
    app.run()

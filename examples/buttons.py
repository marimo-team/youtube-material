import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    btn = mo.ui.button(
        value=0, 
        tooltip="Increase a value on click (Ctrl-U)",
        label='Click me', 
        on_click=lambda d: d + 1, 
        keyboard_shortcut="Ctrl-U"
    )
    return (btn,)


@app.cell
def _(btn):
    btn.value
    return


@app.cell
def _(btn):
    btn
    return


@app.cell
def _(btn):
    btn.center()
    return


if __name__ == "__main__":
    app.run()

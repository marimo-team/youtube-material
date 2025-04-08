import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import time
    return mo, time


@app.cell
def _():
    a = 22
    return (a,)


@app.cell
def _(mo):
    run_btn = mo.ui.run_button()
    run_btn
    return (run_btn,)


@app.cell
def _(a, mo, run_btn, time):
    mo.stop(not run_btn.value, mo.md("Press button to run 🔥"))
    print("""Running big compute again 🤖""")

    time.sleep(2)
    b = a + 1
    return (b,)


@app.cell
def _(b):
    b
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

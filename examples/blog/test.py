import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    import pytest
    return (pytest,)


@app.cell
def _(pytest):
    @pytest.mark.parametrize("a,b,c", [(1, 2, 3), (2, 2, 4), (4, 4, 8)])
    def test_addition(a, b, c):
        assert a + b == c
    return (test_addition,)


@app.cell
def _():
    return


@app.cell
def _():
    def test_basic():
        assert True
    return (test_basic,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

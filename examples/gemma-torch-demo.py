import marimo

__generated_with = "0.13.13"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import sys

    sys.path.append('gemma_pytorch/gemma')
    return


@app.cell
def _():
    from gemma_pytorch.gemma.config import get_model_config
    from gemma_pytorch.gemma.gemma3_model import Gemma3ForMultimodalLM

    import os
    import torch
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

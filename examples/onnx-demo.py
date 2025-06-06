# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "folium==0.19.6",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "onnxruntime==1.22.0",
#     "onnxscript==0.2.7",
#     "polars==1.30.0",
#     "polars-h3==0.5.6",
#     "requests==2.32.3",
#     "torch==2.7.0",
# ]
# ///

import marimo

__generated_with = "0.13.13"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _():
    import torch

    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 128, 5)

        def forward(self, x):
            return torch.relu(self.conv1(x))

    input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

    model = MyModel()

    torch.onnx.export(
        model,                  # model to export
        (input_tensor,),        # inputs of the model,
        "my_model.onnx",        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        dynamo=True             # True or False to select the exporter to use
    )
    return (input_tensor,)


@app.cell
def _(input_tensor):
    input_tensor.shape
    return


@app.cell
def _(response):
    response
    return


@app.cell
def _():
    import requests

    url = "https://github.com/marimo-team/youtube-material/raw/refs/heads/main/my_model.onnx"
    response = requests.get(url)

    with open("my_model.onnx", "wb") as f:
        f.write(response.content)
    return (response,)


@app.cell
def _(np):
    import onnxruntime as rt

    # First we must start a session.
    sess = rt.InferenceSession("my_model.onnx")
    # The name of the input is saved as part of the .onnx file.
    # We are retreiving it because we will need it later.
    input_name = sess.get_inputs()[0].name

    # This code will run the model on our behalf.
    query = "this is an example"
    probas = sess.run(None, {input_name: np.random.random(size=(1, 1, 128, 128)).astype(np.float32)})
    probas[0]
    return


@app.cell
def _(np, slider):
    import polars_h3 as plh3
    import polars as pl

    n = 10_000

    df = pl.DataFrame(
        {
            "lat": np.random.normal(loc=37.7749, scale=0.1, size=n),
            "long": np.random.normal(loc=-122.4194, scale=0.1, size=n),
        }
    ).with_columns(
        plh3.latlng_to_cell(
            "lat",
            "long",
            resolution=slider.value,
            return_dtype=pl.Utf8
        ).alias("h3_cell"),
    ).group_by("h3_cell").len()
    return df, plh3


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 8, 1, label="resolution")
    slider
    return (slider,)


@app.cell
def _(df, plh3):
    plh3.graphing.plot_hex_fills(df, "h3_cell", "len")
    return


if __name__ == "__main__":
    app.run()

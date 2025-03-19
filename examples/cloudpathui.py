# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloudpathlib[all]==0.21.0",
#     "marimo==0.11.22",
#     "polars==1.25.2",
#     "python-dotenv==1.0.1",
# ]
# ///

import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    from cloudpathlib import CloudPath, S3Client, GSClient
    from dotenv import load_dotenv
    import os
    import marimo as mo
    import polars as pl
    from pathlib import Path

    load_dotenv(".env")
    return CloudPath, GSClient, Path, S3Client, load_dotenv, mo, os, pl


@app.cell
def _(GSClient, S3Client, os):
    client = S3Client(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"], 
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], 
    )

    gs_client = GSClient("storage_credentials.json", project="ml-babies")
    return client, gs_client


@app.cell
def _(gs_client):
    cloudpath = gs_client.CloudPath("gs://my-house-data/folder/")
    cloudpath
    return (cloudpath,)


@app.cell
def _(cloudpath, mo):
    file_browser = mo.ui.file_browser(
        initial_path=cloudpath
    )
    return (file_browser,)


@app.cell
def _(file_browser):
    file_browser
    return


@app.cell
def _(file_browser):
    file_browser.value
    return


@app.cell
def _(file_browser, pl):
    pl.concat([
        pl.read_csv(f.path.read_bytes(), separator=";") 
        for f in file_browser.value
    ]).head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi==0.115.12",
#     "httpx==0.28.1",
#     "marimo",
#     "pydantic==2.11.4",
# ]
# ///

import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")

with app.setup:
    from pydantic import BaseModel
    from fastapi import FastAPI, Depends
    from typing import Annotated


    api = FastAPI()


    class Item(BaseModel):
        name: str
        description: str | None = None
        price: float
        tax: float | None = None


@app.function
@api.post("/items/")
async def create_item(item: Item):
    return item


@app.cell
async def _():
    item = Item(name="hello", description="there", price=0.99, tax=0.21)

    await create_item(item=item)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Hold on here!

    This is a pretty interesting "trick". But it is a bit naive to call fastapi routes this way. In particular, you won't be able to capture all the processing that takes place when an actual request is sent through. The above approach might work for simple routes, but let's consider something that is a bit more realistic.
    """
    )
    return


@app.function
async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.function
@api.get("/stuff/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons


@app.cell
async def _():
    await read_items(commons=["hello"])
    return


@app.cell
def _():
    return


@app.cell
def _():
    from fastapi.testclient import TestClient

    client = TestClient(api)
    return (client,)


@app.cell
def _(client):
    client.get("/stuff/").json()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

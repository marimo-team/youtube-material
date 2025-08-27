# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dice==4.0.0",
#     "fastmcp==2.11.3",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")

with app.setup:
    from fastmcp import FastMCP, Client
    import dice

    mcp = FastMCP("MyServer")

    @mcp.tool
    def roll_dice(expression: str) -> str:
        """Whenever the user wants to run hello() on something."""
        return str(dice.roll(expression))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Just to play it on the safe side we can add a few assert statements on the async code here. """)
    return


@app.cell
async def _():
    # Just a few sanity checks
    client = Client(mcp)

    async def call_tool(name: str, **kwargs):
        async with client:
            result = await client.call_tool("roll_dice", kwargs)
            return result.content[0].text

    values = set()
    for i in range(100):
        out = await call_tool("roll_dice", expression="d4")
        eyes = int(out[1])
        values = values.union({eyes})
    assert values == {1, 2, 3, 4}

    out = await call_tool("roll_dice", expression="4d4")
    assert len(out[1:-1].split(",")) == 4
    return call_tool, client


@app.cell
async def _(client):
    async with client: 
        for tool in await client.list_tools():
            print(tool)
    return


@app.cell
async def _(call_tool):
    await call_tool("roll_dice", expression="4d4")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, we can actually run our app in script mode below.""")
    return


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        mcp.run()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
